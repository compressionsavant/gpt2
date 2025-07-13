import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import HfApi
from model import GPT, Config

"""
████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗██████╗ ██╗   ██╗███╗   ██╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗██║   ██║████╗  ██║
   ██║   ██║   ██║██████╔╝██║     ███████║██████╔╝██║   ██║██╔██╗ ██║
   ██║   ██║   ██║██╔══██╗██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║
   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""

class DistributedDataLoader(IterableDataset):
    def __init__(self, B: int, T: int, rank: int, world_size: int, split: str):
        super().__init__()
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.root = "fineweb10B"
        assert split in ["train", "val"]
        shards = os.listdir(self.root)
        shards = sorted([shard for shard in shards if split in shard])
        self.shards = [os.path.join(self.root, shard) for shard in shards]

        if master:
            print(f"using {len(self.shards)} shards for {split} split")
        
        self.reset()

    def reset(self):
        self.shard_pos = 0
        self.tokens = self.load_shard(self.shards[self.shard_pos])
        self.pos = self.B * self.T * self.rank
    
    def load_shard(self, filename: str):
        t = np.load(filename).astype(np.int32)
        pt = torch.from_numpy(t).long()
        return pt

    def __iter__(self):
        while True:
            if self.pos + (self.B * self.T * self.world_size + 1) > len(self.tokens):
                self.shard_pos = (self.shard_pos + 1) % len(self.shards)
                self.tokens = self.load_shard(self.shards[self.shard_pos])
                self.pos = self.B * self.T * self.rank
            
            tok_buf = torch.tensor(self.tokens[self.pos : self.pos + self.B * self.T + 1])
            x = tok_buf[:-1].view(self.B, self.T)
            y = tok_buf[1:].view(self.B, self.T)
            self.pos += self.B * self.T * self.world_size
            x, y = x.to(device), y.to(device)
            yield x, y


# setting up ddp
assert torch.cuda.is_available()
init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master = ddp_rank == 0

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision("high")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# https://arxiv.org/pdf/2005.14165
batch_size = 524288 # 2**19
micro_batch = 64
seq_len = 1024

# no grad accum on 8 gpus 64 * 1024 * 8 = 524288
assert batch_size % (micro_batch * seq_len * ddp_world_size) == 0
grad_accum_steps = batch_size // (micro_batch * seq_len * ddp_world_size)

if master:
    print(f"{grad_accum_steps} grad accum steps")


train_dataset = DistributedDataLoader(micro_batch, seq_len, ddp_rank, ddp_world_size, "train")
val_dataset = DistributedDataLoader(micro_batch, seq_len, ddp_rank, ddp_world_size, "val")
train_loader = iter(train_dataset)
val_loader = iter(val_dataset)

model = GPT(Config(vocab_size=50304))
model.to(device)
model = torch.compile(model, dynamic=True)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

max_steps = int(10e9 // batch_size) # 1 epoch
warmup_steps = int(375e6 // batch_size) # the paper uses 375M tokens for warmup

optim = raw_model.configure_optim(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
warmup = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optim, T_max=max_steps-warmup_steps, eta_min=6e-4*0.1)
scheduler = SequentialLR(optim, [warmup, cosine], milestones=[warmup_steps])

if master:
    writer = SummaryWriter(f"runs/gpt2-124M")

for step in range(max_steps):
    last_step = (step == max_steps - 1)

    if step % 250 == 0:
        model.eval()
        val_dataset.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                x, y = next(val_loader)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_steps
                val_loss_accum += loss.detach()
            
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master:
                writer.add_scalar("val_loss", val_loss_accum, step)
                print(f"step {step} | val_loss {val_loss_accum}")

    model.train()
    optim.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = next(train_loader)
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    scheduler.step()
    torch.cuda.synchronize()
    
    if master:
        writer.add_scalar("train loss", loss_accum, step)
        writer.add_scalar("norm", norm.item(), step)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        print(f"step {step} | train loss {loss_accum} | norm {norm.item()} | lr {scheduler.get_last_lr()[0]}")
        if step % 1000 == 0:
            writer.flush()

    if master and (step % 5000 == 0 or last_step):
        api = HfApi()
        folder = f"step_{step}"
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        checkpoint = {
            "model": raw_model.state_dict(),
            "optim": optim.state_dict(),
            "config": raw_model.config,
            "rng": [cpu_state, cuda_state]
        }
        torch.save(checkpoint, "checkpoint.pth")
        api.upload_file(path_or_fileobj="checkpoint.pth", path_in_repo=f"{folder}/checkpoint.pth", repo_id="compressionsavant/gpt2")
        os.remove("checkpoint.pth")
    dist.barrier()
destroy_process_group()
