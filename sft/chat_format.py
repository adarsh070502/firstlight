"""
The chat format: special tokens, conversation -> (ids, mask), decoding, and generating one assistant reply.

    BOS <|user_start|> What is 2+2? <|user_end|> <|assistant_start|> It is 4. <|assistant_end|> <|user_start|> ...
     0        0          0  0  0  0       0              0             1  1  1          1               0

mask = 1 -> the model is trained to predict this token (assistant text + <|assistant_end|>, so it learns to stop)
"""
import os
# keep tiktoken's downloaded GPT-2 vocab inside the project: its default cache is /tmp, which a reboot wipes,
# and right after a power cut there may be no internet to download it again
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tiktoken_cache"))

import tiktoken
import torch
from torch.nn import functional as F

enc = tiktoken.get_encoding("gpt2")

# GPT-2's tokenizer has ids 0..50256, our model has 50304 rows (padded "nice number"), so 50257..50303 are free
BOS = 50256             # <|endoftext|>: already exists, it started every FineWeb document during pretraining
USER_START = 50257
USER_END = 50258
ASSISTANT_START = 50259
ASSISTANT_END = 50260   # ids above this (50261..50303) are never used

SPECIAL_NAMES = {
    BOS: "<|endoftext|>",
    USER_START: "<|user_start|>",
    USER_END: "<|user_end|>",
    ASSISTANT_START: "<|assistant_start|>",
    ASSISTANT_END: "<|assistant_end|>",
}


def render_conversation(messages):
    """messages: [{"role": "system"/"user"/"assistant", "content": str}, ...]
    returns ids, mask: two lists of the same length, mask 1 = train on this token"""
    # an optional system message is merged into the first user message (same as nanochat)
    if messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            raise ValueError("system message must be followed by a user message")
        merged = {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
        messages = [merged] + list(messages[2:])

    ids, mask = [BOS], [0]
    for i, message in enumerate(messages):
        # user, assistant, user, assistant, ...
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {i} has role {message['role']}, expected {expected}")
        content = enc.encode_ordinary(message["content"]) # ordinary: text like "<|endoftext|>" inside content stays plain text

        if expected == "user":
            tokens = [USER_START] + content + [USER_END]
            ids += tokens
            mask += [0] * len(tokens)
        else:
            ids += [ASSISTANT_START]
            mask += [0] # we insert it ourselves at chat time, the model never has to predict it
            ids += content + [ASSISTANT_END]
            mask += [1] * (len(content) + 1)
    return ids, mask


def render_for_completion(messages):
    """conversation ending with a user message -> ids ending with <|assistant_start|>, ready for generate_reply"""
    assert messages[-1]["role"] == "user"
    ids, _ = render_conversation(messages)
    return ids + [ASSISTANT_START]


def decode(ids):
    # tiktoken only knows ids up to 50256, so special tokens are decoded by hand
    out, run = [], []
    for t in ids:
        t = int(t)
        if t >= BOS:
            if run:
                out.append(enc.decode(run))
                run = []
            out.append(SPECIAL_NAMES.get(t, f"<|unused_{t}|>"))
        else:
            run.append(t)
    if run:
        out.append(enc.decode(run))
    return "".join(out)


def visualize(ids, mask):
    """decoded conversation with the trained (mask=1) parts wrapped in [...]: the Step 2 check"""
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or mask[i] != mask[start]:
            text = decode(ids[start:i])
            out.append(f"[{text}]" if mask[start] else text)
            start = i
    return "".join(out)


@torch.no_grad()
def generate_reply(model, ids, max_new_tokens=256, temperature=0.7, top_k=50, generator=None):
    """ids: the conversation so far, ending with <|assistant_start|>.
    Samples until a special token (normally <|assistant_end|>) and returns the reply's token ids (without it)."""
    device = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    reply = []
    for _ in range(max_new_tokens):
        if x.size(1) >= model.config.block_size:
            break # context is full
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, -1, :].float() # (1, vocab_size)
        logits[:, ASSISTANT_END + 1:] = float('-inf') # 50261..50303 were never trained, never sample them
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            topk_logits, topk_indices = torch.topk(logits / temperature, top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            ix = torch.multinomial(probs, 1, generator=generator)
            next_id = torch.gather(topk_indices, -1, ix) # (1, 1)
        if next_id.item() >= BOS:
            break # end of the reply
        reply.append(next_id.item())
        x = torch.cat((x, next_id), dim=1)
    return reply
