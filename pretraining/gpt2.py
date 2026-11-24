import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, pipeline, set_seed
from dataclasses import dataclass, asdict
import tiktoken
import math
import time
import os
import sys
import inspect
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np


# model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

# sd_hf = model_hf.state_dict()

# for identifier, tensor in sd_hf.items():
#     print(identifier, tensor.shape)


# sd_hf['transformer.wte.weight'][0][:20]


# generator = pipeline("text-generation", model='gpt2')
# set_seed(42)
# generator("Hello I am language model, ", max_length=30, num_return_sequences=5)


# encoder = tiktoken.get_encoding("gpt2")
# encoder.n_vocab


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375M tokens / 524288, the GPT-3 warmup
max_steps = 18751 # 18751 * 524288 = 9.83B tokens = 1 epoch of the train shards; 18750 % 250 == 0, so the last step validates + saves
total_batch_size = 524288 # 2**19, ~0.5M tokens per step
B = 4 # micro-batch size: max that fits in 8GB at T=1024 (B=8 OOMs)
T = 1024 # sequence length, same as GPT-2

log_dir = "log" # checkpoints are saved here: ckpt_last.pt (latest) and ckpt_best.pt (lowest val loss)
init_from = "log/ckpt_last.pt" # None = train from scratch, or a checkpoint path (e.g. "log/ckpt_last.pt") to continue training
ckpt_every = 100 # save ckpt_last.pt every this many steps, so a crash/shutdown loses at most this many steps

@dataclass
class GPTConfig:
    # type annotations make these real dataclass fields, so GPTConfig(**saved_config) works when loading a checkpoint
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50304
    n_embd: int = 768
    block_size: int = T


def get_lr(it):
    if it < warmup_steps:
        # 1) Linear warmup region for warmup_iters steps
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        # 2) if it > lr_decay_iters, flat out return the min_lr
        return min_lr
    # 3) In between warmup and max_steps, cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to zero
    return min_lr + coeff * (max_lr - min_lr)


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # See errata
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B # batch size
        self.T = T # context size
        self.process_rank = process_rank # rank of the current process
        self.num_processes = num_processes # total number of processes
        assert split in {'train', 'val'}, "split must be one of 'train' or 'val'"

        # get the shard filenames
        data_root = "./GPT2/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split '{split}' in '{data_root}'"
        if master_process:
            print(f"found {len(shards)} shards for split '{split}'")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.tok_count = len(self.tokens)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # grab a chunk of tokens of size B * T + 1 (we explained this before)
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # input tensor of size (B * T)
        y = buf[1:].view(B, T)  # target tensor of size (B * T), right-shifted 1 position
        # advance position in data tensor
        # NEW: Jump to the next batch, taking into account the number of processes
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        # NEW: Check if our parallel-process-aware jump would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # reset position to beginning of data, w.r.t. process rank
            self.current_position = self.B * self.T * self.process_rank
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.config = config

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size() # 32, 1024, 768

        qvk = self.c_attn(x) # x.shape=(32, 1024, 768) and qvk.shape = (32, 1024, 2304) (In linear layer: (32, 1024, 768) @ (32, 768, 2304))

        q, k, v = qvk.split(self.n_embd, dim=2) # q,k,v shapes will be (32, 1024, 768)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
        
class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        # Linear weights distributed with mean 0 and std 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 1 / sqrt(N) scaling fo the std dev
                # 2 * n_layer because we have attention *and* mlp per block (blocks counted by n_layer)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # Biases explicitly initialized to 0
                torch.nn.init.zeros_(module.bias)
        # Embedding weights distributed with mean 0 and std 0.02
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        pos_emb = self.transformer.wpe(pos) # B,T,768
        tok_emb = self.transformer.wte(idx) # B,T,768

        x = tok_emb + pos_emb # B,T,768

        for block in self.transformer.h:
            x = block(x) # B,T,768

        x = self.transformer.ln_f(x) # B,T,768

        logits = self.lm_head(x) # B,T,vocab_size

        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss # B,T,vocab_size


    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, but all biases and layer-norms don't.
        # Effectively splitting the parameters into two groups: decayable and non-decayable
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # 1D tensors (layer norms, biases) are not decayed
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # Essentially checking here if the option 'fused' is available in the AdamW optimizer (depends on PyTorch version)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    assert torch.cuda.is_available(), "For DDP, cuda is required"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))

    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")


torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

assert total_batch_size % (B * T * ddp_world_size) == 0, "Batch size must be divisible by (micro-batch size * sequence length * ddp_world_size)."
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # 32 micro-batches making up 1 macro-batch

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# TF32 tensor float precision for matmuls
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)
raw_model = model # the plain GPT, before torch.compile / DDP wrap it: used for checkpoints and the optimizer

use_compile = True # Turn this on if you have a CUDA-capable GPU and Linux
if use_compile and os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # compile model to TorchScript -> speed + memory savings
else:
    print("[!] Not running Linux - Skipping platform-unsupported torch.compile()")


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)


os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

def log(msg):
    # print to the screen and also append to log/log.txt ("a" mode, so a resumed run keeps the earlier history)
    print(msg)
    if master_process:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

def save_checkpoint(path, step, val_loss, best_val_loss):
    checkpoint = {
        # for inference: weights + the config to rebuild the model
        'model': raw_model.state_dict(),
        'config': asdict(raw_model.config),
        # for continuing training: optimizer moments, where we are in the schedule and in the data
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_shard': train_loader.current_shard,
        'train_position': train_loader.current_position,
    }
    # write to a temp file first, so a crash mid-save never corrupts the previous checkpoint
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno()) # force it onto the disk now, so a power cut can't leave a file of zeros behind
    os.replace(tmp_path, path)

start_step = 0
best_val_loss = float('inf')
val_loss = None # last measured val loss (ckpt_last.pt saved between validations stores this one)
if init_from is not None:
    ckpt = torch.load(init_from, map_location='cpu', weights_only=True)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer']) # moves the AdamW state onto the params' device
    start_step = ckpt['step'] + 1
    best_val_loss = ckpt['best_val_loss']
    val_loss = ckpt['val_loss']
    # put the train loader back where it stopped, so we don't re-train on the same tokens
    train_loader.current_shard = ckpt['train_shard']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    train_loader.current_position = ckpt['train_position'] + B * T * ddp_rank
    if master_process:
        log(f"resumed from {init_from}: continuing at step {start_step}, best val loss so far {best_val_loss:.4f}")
    del ckpt


enc = tiktoken.get_encoding('gpt2')
for step in range(start_step, max_steps):
    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum+=loss.detach()
        if ddp:
            # DDP will take care of averaging the loss across all GPUs
            # DDP will do so only after the last micro-batch in each accumulation cycle
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        # Average the macro-batch loss across all GPUs, 
        # after having done that, synchronize this average loss to be the same value set across all GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    if "cuda" in device:
        torch.cuda.synchronize()

    t1 = time.time()

    dt = (t1-t0) 
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps 
    tokens_per_sec = tokens_processed // dt
    log(f"epoch {step}, loss: {loss_accum.item()}, lr: {lr:.4e} | norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

    if step % 250 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss = val_loss_accum.item()
            log(f"validation loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(log_dir, "ckpt_best.pt"), step, val_loss, best_val_loss)

    # latest state, saved every ckpt_every steps and at every validation
    if master_process and (step % ckpt_every == 0 or step % 250 == 0):
        t_save = time.time()
        save_checkpoint(os.path.join(log_dir, "ckpt_last.pt"), step, val_loss, best_val_loss)
        log(f"saved ckpt_last.pt at step {step} ({time.time() - t_save:.1f}s)")

    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # seed the number generator for reproducibility
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            log(f"rank {ddp_rank} sample {i}: {decoded}")

    # if step == 5:
    #     import sys; sys.exit(0)


# load a checkpoint for inference: only the weights + config are needed, not the optimizer state
ckpt = torch.load(os.path.join(log_dir, "ckpt_best.pt"), map_location='cpu', weights_only=True)
model = GPT(GPTConfig(**ckpt['config']))
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()
print(f"loaded checkpoint from step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}")
del ckpt


model.eval()
num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("King:")
tokens = torch.tensor(tokens)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to(device)
print(f"x shape is {x.shape}")

while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)

        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


# epoch 1845, loss: 3.6703579425811768, lr: 5.9479e-04 | norm: 0.2976, dt: 32356.32ms, tok/sec: 16203.00

# epoch 2472, loss: 3.5692543983459473, lr: 5.8745e-04 | norm: 0.3516, dt: 33114.25ms, tok/sec: 15832.00