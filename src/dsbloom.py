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
CTXLEN = 1024
CTXLEN = 2048

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

# get the deepspeed configuration
def get_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": GRADACC,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.stage,
        },
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


