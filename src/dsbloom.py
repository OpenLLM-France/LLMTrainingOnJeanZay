import os
from time import time
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import transformers
from transformers import AutoModelForCausalLM, TrainingArguments
import idr_torch  # IDRIS library to make distribution on JZ easier

STAGE = 1
LR = 2e-05

# TODO:
# - [X] new model from config with max vocab from megatron tokenizer (32000 tokens)
# - [X] add LR scheduler (pass to deepspeed config)
# - [X] init model weights: A very low standard initialization sqrt(0.3333/NHIDDEN) - in 104B-en used sqrt(0.4/NHIDDEN) see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md
# - [ ] save model+opt+scheduler+... checkpoints

# pending questions:
# - really should pack data to 2048 ? Does not seem to make a big diff with keeping natural sentences... ?
# - sample dataset randomly directly from parquet files ? If I pack to 2048, then shuffle, then it'll be hard to progressively increase context length; .parquet are compressed, and must be read sequentially and decompressed; while arrow files are memory map and enable random access in O(1) ! (pyarrow.ipc.open_file())
# - 1bitLamb could help me remove gradient checkpointing?
# - continuous training from bloom-7b, with 5%-mag pruning when loss stagne or duplicating 1 layer ?

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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/gpfsdswork/dataset/HuggingFace/wikipedia/fr')
    parser.add_argument('--model_dir', type=str, default='/gpfsdswork/dataset/HuggingFace_Models/')
    parser.add_argument('--model_name', type=str, default='bigscience/bloom-7b1')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--stage', type=int, default=STAGE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    return args

# get the deepspeed configuration see https://www.deepspeed.ai/docs/config-json/#batch-size-related-parameters
def get_ds_config(args):
    ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": args.stage,
                },
            "flops_profiler": {
                "enabled": False,
                "profile_step": 1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
                "output_file": None,
                },
            "optimizer": {
                "type": "FusedLamb",
                "params": {
                    "lr": LR,
                    }
                },
            "gradient_clipping": 1.0,
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": 250000000000,
                    "warmup_num_steps": 1000
                    }
                },
            "wall_clock_breakdown": False,
            "gradient_accumulation_steps": 8,
            "train_micro_batch_size_per_gpu": 1,

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

def train_loop(model, train_dataloader):

    for i, data in enumerate(train_dataloader):
        inputs = {'input_ids' : data.to(model.local_rank)}
        xx = inputs['input_ids']
        inputs['labels'] = xx.clone()
        out = model(**inputs)
        loss = out.loss
        print("LOSS",loss.item())
        model.backward(loss)
        model.step()
        print("datainputVRAM",xx.shape, idr_torch.rank, torch.cuda.memory_allocated())

    return model

class LucieDataset(torch.utils.data.Dataset):
    def __init__(self, toker):
        self.toker = toker
        self.f = open("/gpfswork/rech/knb/uyr14tk/home/openllmfr/alldata/all.txt")
        self.fichs=[]
        self.idx=[]
        with open("/gpfswork/rech/knb/uyr14tk/home/openllmfr/alldata/idx.txt") as f:
            for l in f:
                if l.startswith("FFNOM__ "):
                    ss=l.split(" ")
                    self.fichs.append(ss[1])
                    self.idx.append(int(ss[2]))
        print("NFICHS",len(self.fichs))

    def __len__(self):
        return len(self.fichs)

    def __getitem__(self, i):
        ii = self.idx[i]
        self.f.seek(ii)
        s=self.f.readLine()
        x=self.toker.batch_encode_plus([s], return_tensors="pt", padding=True)
        return x['input_ids']
 
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Configure the tokenizer to ensure padding is done right
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    dataset = LucieDataset(tokenizer)

    # Need sampler for Distributed Training
    train_sampler = DistributedSampler(
        dataset,
        shuffle=False,
        num_replicas=idr_torch.world_size,
        rank=idr_torch.rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=2,
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # Prepare model, dataset... for distributed training with deepspeed
    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )


    # 1 epoch
    start_epoch = time()
    train_sampler.set_epoch(0)
    model = train_loop(model, train_dataloader)
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


