import os
from time import time
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
from datasets import load_from_disk
from tqdm import tqdm
from deepspeed.sequence.layer import DistributedAttention
import idr_torch  # IDRIS library to make distribution on JZ easier

GRADACC = 64
CTXLEN = 2048
CTXLEN = 1024

print("deepspeedversion",deepspeed.__version__)

# Initialize Distributed Training
deepspeed.init_distributed(
        dist_backend="nccl",
        init_method="env://",
)
print("process",idr_torch.local_rank)
torch.cuda.set_device(idr_torch.local_rank)
DEVICE = torch.device("cuda")

# Set Seed for Reproducibility
torch.manual_seed(53)
torch.cuda.manual_seed(53)

# Avoid tokenizers warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for DistributedAttention
class LocAtt(torch.nn.Module):
    def forward(self, query_layer, key_layer, value_layer, num_heads, head_dim, layer_past, use_cache, alibi, beta, inv_norm_factor, attention_mask, head_mask, attention_dropout, _merge_heads):
        batch_size, q_length, _, _ = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size * num_heads, q_length, head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * num_heads, head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_heads, q_length, head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)
        _, _, kv_length = key_layer.shape
        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=beta,
            alpha=inv_norm_factor,
        )
        attention_scores = matmul_result.view(batch_size, num_heads, q_length, kv_length)
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16: attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
        attention_probs = attention_dropout(attention_probs)
        if head_mask is not None: attention_probs = attention_probs * head_mask
        attention_probs_reshaped = attention_probs.view(batch_size * num_heads, q_length, kv_length)
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)
        context_layer = _merge_heads(context_layer)
        return context_layer

locatt = LocAtt()

class DistAttBloom(transformers.models.bloom.modeling_bloom.BloomAttention):
    def dattinit(self):
        pass

    def forward(self, hidden_states, residual, alibi, attention_mask, layer_past=None, head_mask=None, use_cache=False, output_attentions=False):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        context_layer = locatt.forward(query_layer, key_layer, value_layer, )
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)])
        else: output_tensor = self.dense(context_layer)
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        outputs = (output_tensor, present)
        if output_attentions: outputs += (attention_probs,)
        return outputs


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Check IDRIS documentation to see which datasets and models are available on DSDIR
    parser.add_argument('--data_path', type=str, default='/gpfsdswork/dataset/HuggingFace/wikipedia/fr')
    parser.add_argument('--model_dir', type=str, default='/gpfsdswork/dataset/HuggingFace_Models/')
    parser.add_argument('--model_name', type=str, default='bigscience/bloom-7b1')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    return args

# get the deepspeed configuration see https://www.deepspeed.ai/docs/config-json/#batch-size-related-parameters
def get_ds_config(args):
    ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": GRADACC,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": args.stage,
                },
            "flops_profiler": {
                "enabled": True,
                "profile_step": 1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
                "output_file": None,
                },
            # "activation_checkpointing": {
            #     "partition_activations": False,
            #     "cpu_checkpointing": False,
            #     "contiguous_memory_optimization": False,
            #     "number_checkpoints": None,
            #     "synchronize_checkpoint_boundary": False,
            #     "profile": False
            #     },
            }
    return ds_config

# print only on rank 0 for cleaner output
def print_rank_0(*args, **kwargs):
    if idr_torch.rank == 0:
        print(*args, **kwargs)

def train_loop(model, tokenizer, train_dataloader, criterion, optimizer):
    model.train()
    # tqdm for a nice progress bar, allow only on rank 0 for cleaner output
    loop = tqdm(train_dataloader, disable=(idr_torch.rank != 0))

    for i, data in enumerate(loop):
        optimizer.zero_grad()

        inputs = tokenizer.batch_encode_plus(
            data['text'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTXLEN,
        ).to(DEVICE)
        xx = inputs['input_ids']
        print("datainputVRAM",xx.shape, idr_torch.rank, torch.cuda.memory_allocated())
        inputs['labels'] = xx.clone()
        out = model(**inputs)
        loss = out.loss
        print("LOSS",loss.item())
        model.backward(loss)
        model.step()

        # print next to progress bar
        loop.set_postfix(loss=loss.item())

    loop.close()
    return model


def main(args):
    # Initialize Datasets
    dataset = load_from_disk(args.data_path)

    # Need sampler for Distributed Training
    train_sampler = DistributedSampler(
        dataset['train'],
        shuffle=True,
        num_replicas=idr_torch.world_size,
        rank=idr_torch.rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'],
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # test_sampler = DistributedSampler(
    #     dataset['test'],
    #     shuffle=True,
    #     num_replicas=idr_torch.world_size,
    #     rank=idr_torch.rank
    # )
    # test_dataloader = torch.utils.data.DataLoader(
    #     dataset['test'],
    #     sampler=test_sampler,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     pin_memory=True,
    #     prefetch_factor=2,
    # )

    # Initialize Model and Tokenizer
    ds_config = get_ds_config(args)
    # Next line enable smart loading (zero.init()) (necessary for very big models)
    _ = TrainingArguments(output_dir="./", deepspeed=ds_config)
    model_path = os.path.join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Configure the tokenizer to ensure padding is done right
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    # model.gradient_checkpointing_enable()

    # Initialize Optimizer and Criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
   
    # Prepare model, dataset... for distributed training with deepspeed
    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), optimizer=optimizer, config=ds_config
    )


    # 1 epoch
    start_epoch = time()
    model = train_loop(model, tokenizer, train_dataloader, criterion, optimizer)
    print_rank_0(
        f"Duration: {(time() - start_epoch):.3f}s "
    )
    print(
        f"Max memory allocated for GPU {idr_torch.rank}:",
        f"{torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3:.1f} GB"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)


