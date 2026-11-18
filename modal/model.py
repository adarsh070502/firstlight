"""
Firstlight-124M (GPT-2 architecture, pretrained from scratch on FineWeb-EDU, then SFT on smol-smoltalk), for the demo app (app.py).

Weights are downloaded from the model repo (REPO_ID) on first use and cached by huggingface_hub:
    base/model.safetensors   the pretrained model (continues text)
    chat/model.safetensors   the SFT model (chat assistant)
    config.json              n_layer, n_head, n_embd, vocab_size, block_size + the special token ids

Same network as GPT2/sft/model.py, with two changes for serving:
- a KV cache: each new token reuses the keys/values of the earlier ones, so one token costs one position's
  forward pass instead of the whole context's (needed on a CPU Space)
- no unused `bias` buffer (the causal mask comes from scaled_dot_product_attention)

Chat format (GPT2/sft/chat_format.py):
    BOS <|user_start|> question <|user_end|> <|assistant_start|> answer <|assistant_end|> <|user_start|> ...
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file

REPO_ID = "adarsh-007/firstlight-124m"
# a folder with the files, to skip the download (optional); otherwise they come from the Hub and are cached
LOCAL_DIR = os.environ.get("GPT2_MODEL_DIR")

enc = tiktoken.get_encoding("gpt2")

# GPT-2's tokenizer has ids 0..50256; the model has 50304 rows, so 50257..50303 were free for the chat tokens
BOS = 50256             # <|endoftext|>
USER_START = 50257
USER_END = 50258
ASSISTANT_START = 50259
ASSISTANT_END = 50260
REAL_VOCAB = 50257      # base model: ids >= this were never trained, never sample them


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50304
    n_embd: int = 768
    block_size: int = 1024


class KVCache:
    """The k and v of every token the model has already seen, for every layer.
    self.k = a Python list of 12 tensors, one per layer: self.k[i] = the keys made by layer i.
    Each tensor is (B, n_head, T, head_size) = (1, 12, T, 64), T = tokens seen so far. self.v is the same for the values."""

    def __init__(self, n_layer):
        self.k = [None] * n_layer  # None = this layer has stored nothing yet
        self.v = [None] * n_layer

    def get_pos(self):
        # how many tokens are stored = the position of the next token
        # every layer stores the same tokens, so the count of layer 0 is the count of all of them
        if self.k[0] is None:
            return 0
        return self.k[0].size(2)  # (B, n_head, T, head_size) -> T

    def add(self, layer_idx, k, v):
        # k, v = the NEW tokens' keys and values from layer layer_idx, each (B, n_head, new tokens, head_size)
        if self.k[layer_idx] is not None:
            # glue the new ones after the old ones along dim 2 (the token dim): (1, 12, 3, 64) + (1, 12, 1, 64) -> (1, 12, 4, 64)
            k = torch.cat((self.k[layer_idx], k), dim=2)
            v = torch.cat((self.v[layer_idx], v), dim=2)
        self.k[layer_idx] = k
        self.v[layer_idx] = v
        return k, v  # the keys and values of ALL tokens so far (old + new)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.layer_idx = layer_idx  # KV cache: which layer this is, so it uses kv_cache.k[layer_idx] and kv_cache.v[layer_idx]

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()  # T = the NEW tokens only: the whole prompt in the first step, then 1
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # KV cache: save the new tokens' k, v and get back the k, v of ALL tokens. q stays the new tokens only
        if kv_cache is not None:
            k, v = kv_cache.add(self.layer_idx, k, v)
        # KV cache: the mask. q and k have the same tokens (first step, or no cache): the normal causal mask.
        # 1 new token: every key is in its past, so it may look at all of them -> no mask
        # (is_causal=True with 1 query and many keys would let it see only the FIRST key)
        assert q.size(2) == k.size(2) or q.size(2) == 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(q.size(2) == k.size(2)))
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
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)  # KV cache: pass the layer number on
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln_1(x), kv_cache)  # KV cache: pass the cache on
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),  # KV cache: block i gets layer_idx = i
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in training

    def forward(self, idx, kv_cache=None):
        B, T = idx.size()  # T = the NEW tokens only
        # KV cache: the new tokens come after the cached ones, so their positions start at the number of cached tokens.
        # Read it BEFORE the blocks run: layer 0 adds the new tokens to the cache, after that the count is bigger
        pos_start = 0 if kv_cache is None else kv_cache.get_pos()
        assert pos_start + T <= self.config.block_size, f"Cannot forward sequence of length {pos_start + T}, block size is only {self.config.block_size}"
        pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, kv_cache)  # KV cache: every block gets the same cache, each uses its own layer's k, v
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)


def model_files(name):
    names = ["config.json", f"{name}/model.safetensors"]
    if LOCAL_DIR:
        return [Path(LOCAL_DIR) / n for n in names]
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(REPO_ID, n)) for n in names]


def load_model(name, device="cpu"):
    """name: "base" (pretrained) or "chat" (SFT)"""
    config_path, weights_path = model_files(name)
    config = json.loads(config_path.read_text())
    model = GPT(GPTConfig(**{k: config[k] for k in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size")}))
    state = load_file(weights_path)
    # lm_head.weight isn't stored: it is the same tensor as transformer.wte.weight (weight tying)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)
    return model.to(device).eval()


# ----------------------------------------------------------------------------- chat format + sampling

def render_chat(messages):
    """[{"role": "user"/"assistant", "content": str}, ...] ending with a user message -> ids ending with
    <|assistant_start|>, ready for the model's reply"""
    ids = [BOS]
    for m in messages:
        content = enc.encode_ordinary(m["content"])
        if m["role"] == "user":
            ids += [USER_START] + content + [USER_END]
        else:
            ids += [ASSISTANT_START] + content + [ASSISTANT_END]
    return ids + [ASSISTANT_START]


@torch.no_grad()
def stream(model, ids, max_new_tokens=300, temperature=0.7, top_k=50, chat=True):
    """Yields the text generated so far after every new token.
    chat=True: stops at <|assistant_end|> (or any special token). chat=False (base model): stops at <|endoftext|>."""
    device = next(model.parameters()).device
    block_size = model.config.block_size
    ids = ids[-(block_size - max_new_tokens):]   # keep the newest tokens if the prompt is too long
    kv_cache = KVCache(model.config.n_layer)      # empty: nothing seen yet
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x, kv_cache)                   # prefill: the whole prompt once, fills the cache
    new = []
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :].float()
        # chat model: only the 4 chat tokens are allowed above 50256. base model: nothing above 50256
        logits[:, (ASSISTANT_END + 1) if chat else REAL_VOCAB:] = float('-inf')
        if temperature == 0:
            next_id = logits.argmax(dim=-1).item()
        else:
            topk_logits, topk_idx = torch.topk(logits / temperature, top_k, dim=-1)
            next_id = topk_idx[0, torch.multinomial(F.softmax(topk_logits, dim=-1), 1)].item()
        if next_id >= BOS:
            break                                 # <|assistant_end|> (chat) or <|endoftext|> (base): done
        new.append(next_id)
        yield enc.decode(new)
        if kv_cache.get_pos() >= block_size:
            break                                 # context full
        logits = model(torch.tensor([[next_id]], device=device), kv_cache)   # decode: ONLY the new token
