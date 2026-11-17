"""
HellaSwag eval of the SFT model, the same method as Andrej's build-nanogpt hellaswag.py.
    python hellaswag.py                    my SFT model (log/ckpt_best.pt), all 10,042 examples (~3 min)
    python hellaswag.py --limit 500        quick test on the first 500 examples
    python hellaswag.py --ckpt other.pt    any other checkpoint of my GPT

Each example = a context + 4 possible endings, only 1 is right. Random guessing = 25%.
    "A man is sitting on a roof. He..."   1) "is using wrap to wrap a pair of skis."  2) "starts pulling up roofing on a roof."  ...
The model doesn't choose by answering. We put each ending after the context, compute the average loss of the
ending's tokens, and the ending with the LOWEST loss (= the one the model finds most likely) is its pick.
This is "acc_norm" (average, so long endings aren't punished).
The SFT model is scored as plain text, without chat tokens: it checks how much pretraining knowledge survived SFT.
"""
import argparse
import os

import torch
from torch.nn import functional as F
from datasets import load_dataset

from chat_format import enc
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high") # TF32


def render_example(example):
    """4 rows = context + each ending, and a mask with 1 on the ending tokens (only those are scored)"""
    ctx_tokens = enc.encode(example["ctx"])
    tok_rows, mask_rows = [], []
    for end in example["endings"]:
        end_tokens = enc.encode(" " + end) # " " because the ending continues the sentence
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    # the endings have different lengths: pad to the longest (the padding has mask 0, so it doesn't count)
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    return tokens, mask, int(example["label"])


def pick_ending(logits, tokens, mask):
    # loss of every token given the tokens before it (logits at position t predict token t+1)
    losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), reduction="none")
    losses = losses.view(4, -1)
    ending_mask = mask[:, 1:]
    avg_loss = (losses * ending_mask).sum(dim=1) / ending_mask.sum(dim=1) # average over the ending's tokens only
    return avg_loss.argmin().item() # most likely ending


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.path.join(HERE, "log", "ckpt_best.pt"))
parser.add_argument("--limit", type=int, default=None, help="only the first N examples (quick test)")
args = parser.parse_args()

examples = list(load_dataset("Rowan/hellaswag", split="validation")) # 10,042 examples
if args.limit:
    examples = examples[:args.limit]

model, _ = load_model(args.ckpt, device)
model.eval()

correct = 0
with torch.no_grad():
    for i, example in enumerate(examples):
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        logits, _ = model(tokens)
        correct += int(pick_ending(logits.float(), tokens, mask) == label)
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(examples)}, acc_norm so far {correct / (i + 1):.4f}", flush=True)

print(f"HellaSwag acc_norm: {correct}/{len(examples)} = {correct / len(examples):.4f}   (random = 0.25)  [{args.ckpt}]")
