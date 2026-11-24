"""
GPT-2 model, copied from ../pretraining/gpt2.py (that file can't be imported: it starts training).
Only change for SFT: cross_entropy uses ignore_index=-1, so targets set to -1 (user tokens, padding) give no loss.
"""
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    # type annotations make these real dataclass fields, so GPTConfig(**saved_config) works when loading a checkpoint
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50304
    n_embd: int = 768
    block_size: int = 1024


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

        # unused (flash attention below does the masking), kept so the pretrained state_dict loads with strict=True
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()

        qvk = self.c_attn(x) # (B, T, 3*C)

        q, k, v = qvk.split(self.n_embd, dim=2) # each (B, T, C)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

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
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        pos_emb = self.transformer.wpe(pos) # T,768
        tok_emb = self.transformer.wte(idx) # B,T,768

        x = tok_emb + pos_emb # B,T,768

        for block in self.transformer.h:
            x = block(x) # B,T,768

        x = self.transformer.ln_f(x) # B,T,768

        logits = self.lm_head(x) # B,T,vocab_size

        loss=None
        if targets is not None:
            # SFT: targets == -1 (user tokens, padding) are skipped, the mean is over the assistant tokens only
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss # B,T,vocab_size


    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D params (matmul weights, embeddings) get weight decay, 1D params (biases, layernorms) don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def load_model(path, device):
    # works for the pretraining checkpoints and the SFT ones: both store 'model' (weights) + 'config'
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'])
    model.to(device)
    return model, ckpt
