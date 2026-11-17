"""
Sentence completion with the GPT-2 (124M) trained by gpt2.py.

Usage (from anywhere):
    python GPT2/inference.py "The best way to learn a new language is"
    python GPT2/inference.py "Photosynthesis is" --num_samples 5 --temperature 0.8
    python GPT2/inference.py "Once upon a time" --no_stop          # don't stop at the end of the sentence
    python GPT2/inference.py                                        # interactive: type prompts, empty line quits

The model classes are copied from gpt2.py because importing gpt2.py would start training.
"""
import argparse
import os
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Datascience (gpt2.py was run from here)
DEFAULT_CKPT = os.path.join(ROOT, "log", "ckpt_best.pt")

GPT2_VOCAB = 50257  # real tokenizer vocab; the model's 50304 is padded for speed
EOT = 50256         # <|endoftext|>


# ----------------------------------------------------------------------------- model (same as gpt2.py)

@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50304
    n_embd: int = 768
    block_size: int = 1024


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # unused at inference (SDPA does the masking), but it's in the checkpoint's state_dict
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in training

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)


# ----------------------------------------------------------------------------- loading + sampling

def load_model(ckpt_path, device):
    # mmap: the file also holds the AdamW state (~1 GB) which we don't need; this avoids reading it all into RAM
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'])
    info = f"step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}"
    del ckpt
    model.to(device).eval()
    return model, info


def ends_sentence(text):
    t = text.rstrip(' "\'')
    return '\n' in text or t.endswith(('.', '!', '?'))


@torch.no_grad()
def complete(model, enc, prompt, device, num_samples=3, max_new_tokens=60,
             temperature=1.0, top_k=50, stop_at_sentence=True, generator=None):
    prompt_ids = enc.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)
    done = torch.zeros(num_samples, dtype=torch.bool, device=device)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]  # never feed more than the context length
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            logits = model(x_cond)
        logits = logits[:, -1, :].float()
        logits[:, GPT2_VOCAB:] = float('-inf')  # padded ids 50257..50303 aren't real tokens
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=generator)
        next_tok = torch.gather(topk_idx, -1, ix)  # (num_samples, 1)
        next_tok[done] = EOT                       # finished rows just get padding
        x = torch.cat((x, next_tok), dim=1)

        for i in range(num_samples):
            if done[i]:
                continue
            tok = next_tok[i].item()
            if tok == EOT or (stop_at_sentence and ends_sentence(enc.decode([tok]))):
                done[i] = True
        if done.all():
            break

    completions = []
    for i in range(num_samples):
        new_ids = x[i, len(prompt_ids):].tolist()
        if EOT in new_ids:
            new_ids = new_ids[:new_ids.index(EOT)]
        text = enc.decode(new_ids)
        if stop_at_sentence and '\n' in text:
            text = text[:text.index('\n')]
        completions.append(text)
    return completions


def main():
    p = argparse.ArgumentParser(description="Sentence completion with your trained GPT-2 124M")
    p.add_argument("prompt", nargs="?", help="start of the sentence; omit for interactive mode")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0, help="<1 = safer/more repetitive, >1 = more random")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--no_stop", action="store_true", help="keep generating past the end of the sentence")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')
    model, info = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")
    print(f"loaded {args.ckpt} ({info}) on {device}")

    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)

    def run(prompt):
        outs = complete(model, enc, prompt, device, num_samples=args.num_samples,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_k=args.top_k, stop_at_sentence=not args.no_stop, generator=gen)
        for i, out in enumerate(outs, 1):
            print(f"[{i}] {prompt}\033[1m{out}\033[0m")

    if args.prompt:
        run(args.prompt.rstrip())  # a trailing space becomes its own token and hurts GPT-2's predictions
        return

    print("type the start of a sentence (empty line or Ctrl-D to quit)")
    while True:
        try:
            prompt = input("\n> ").rstrip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
