"""
SFT: turn the pretrained GPT-2 124M into a chat assistant (see docs/interview_notes.md).
    python sft.py    continue from log/ckpt_last.pt if it exists, else start from ../pretraining/log/ckpt_best.pt
                     (so after a crash / power cut / Ctrl+C, just run the same command again)
To start over from the pretrained model: delete log/ckpt_last.pt (or move the whole log/ folder away, to keep the old model).
Writes to log/: log.txt (everything printed), metrics.csv (losses, for plots), ckpt_best.pt (weights with the
lowest val loss), ckpt_last.pt (weights + optimizer, to continue). Needs data/ from prepare_data.py.
"""
import os
# every batch has a different length: expandable segments stop CUDA memory fragmentation (must be set before torch uses CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from dataclasses import asdict

import numpy as np
import torch

from chat_format import BOS, enc, generate_reply, render_for_completion
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__)) # all paths relative to this file, so it runs from any folder
pretrained_ckpt = os.path.join(HERE, "..", "pretraining", "log", "ckpt_best.pt")
data_dir = os.path.join(HERE, "data")
log_dir = os.path.join(HERE, "log")

max_lr = 5e-5          # ~ where pretraining ended (6e-5), 12x lower than pretraining's peak
warmup_steps = 50
weight_decay = 0.0
target_examples_per_step = 32 # conversations per optimizer step (nanochat's name and value)
B = 4                  # micro-batch size: conversations per forward/backward (nanochat calls it device_batch_size, also 4)
assert target_examples_per_step % B == 0, "target_examples_per_step must be divisible by B"
grad_accum_steps = target_examples_per_step // B # 32 // 4 = 8 micro-batches, then one optimizer step
num_epochs = 1         # how many times we go through all 403,748 train conversations (1 = each one seen once, like nanochat)
                       # used in: max_steps = steps_per_epoch * num_epochs = 12,617 * 1 = 12,617 steps
eval_every = 250       # measure the val loss every 250 steps
val_loss_steps = 125   # val loss on 125 micro-batches x 4 = 500 of the 21,191 test conversations (~300k tokens), like
                       # Andrej's val_loss_steps = 20: the full test set (13.2M tokens) every 250 steps would slow training down a lot.
                       # val_loader.reset() before every check -> always the same 500, so the val losses are comparable.
sample_every = 500     # log the model's answers to sample_prompts
ckpt_every = 500       # save log/ckpt_last.pt, so a crash loses at most this many steps (~11 min)

sample_prompts = [
    "What is photosynthesis?",
    "Hi! Can you give me three tips for studying for an exam?",
]

device = "cuda"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision('high')

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

def log(msg):
    # print to the screen and also append to log/log.txt ("a" mode, so a resumed run keeps the earlier history)
    print(msg, flush=True)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# log/metrics.csv: one row per step for the loss plot (../huggingface/plot_sft_loss.py). step -1 = the val loss before SFT
metrics_file = os.path.join(log_dir, "metrics.csv")
METRIC_COLUMNS = ["step", "train_loss", "val_loss", "lr", "norm", "tok_per_sec"]

def reset_metrics(keep_until_step=None):
    # fresh run: an empty metrics.csv. resumed run: drop the rows after the checkpoint, those steps are trained again
    rows = []
    if keep_until_step is not None and os.path.exists(metrics_file):
        with open(metrics_file) as f:
            # a power cut can leave zero bytes / a half-written last line: keep only complete rows ("\n"-terminated)
            lines = f.read().replace("\x00", "").splitlines(keepends=True)[1:]
        rows = [line for line in lines if line.endswith("\n") and int(line.split(",")[0]) <= keep_until_step]
    with open(metrics_file, "w") as f:
        f.write(",".join(METRIC_COLUMNS) + "\n")
        f.writelines(rows)

def record(step, **values):
    row = [str(step)] + [f"{values[c]:.6g}" if c in values else "" for c in METRIC_COLUMNS[1:]]
    with open(metrics_file, "a") as f:
        f.write(",".join(row) + "\n")


def load_tokens(filename):
    # the same as in pretraining (gpt2.py)
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, split):
        self.B = B # batch size: conversations per micro-batch
        assert split in {'train', 'test'}, "split must be one of 'train' or 'test'"

        self.ids = load_tokens(os.path.join(data_dir, f"{split}_ids.npy"))
        self.mask = load_tokens(os.path.join(data_dir, f"{split}_mask.npy"))
        # Where does each conversation start?
        # In chat_format.py, render_conversation begins every conversation with:  ids, mask = [BOS], [0]
        # So every conversation starts with a BOS token, and BOS is put only there.
        # -> find the BOS tokens in self.ids = find where the conversations start.
        #
        # Example: 3 conversations of 4, 3 and 5 tokens, back to back in self.ids (t = any normal token):
        #   position:   0    1  2  3    4    5  6    7    8  9  10 11
        #   self.ids:   BOS  t  t  t    BOS  t  t    BOS  t  t  t  t
        #               |-- conv 0 --|  |- conv 1 -| |--- conv 2 ----|
        #
        #   self.ids == BOS  -> True where a conversation starts:  [T F F F T F F T F F F F]
        #   .nonzero()       -> the positions of the Trues:        [[0], [4], [7]]   (shape (3, 1): one row per True)
        #   .view(-1)        -> flatten to 1D:                     [0, 4, 7]
        #   .tolist()        -> tensor to a plain Python list:     [0, 4, 7]   (one number per conversation, small,
        #                                                                       so a list is fine; tokens stay a tensor)
        self.starts = (self.ids == BOS).nonzero().view(-1).tolist()

        # The last conversation has no BOS after it to mark its end, so add the end of the array: [0, 4, 7, 12]
        self.starts.append(len(self.ids))

        # Now each conversation = from its start up to the next start:
        #   conversation i = self.ids[self.starts[i] : self.starts[i + 1]]
        #   conv 0 = self.ids[0:4],   conv 1 = self.ids[4:7],   conv 2 = self.ids[7:12]
        # Its length = next start - its start:  [4 - 0, 7 - 4, 12 - 7] = [4, 3, 5]
        # range(len(self.starts) - 1): starts has 4 numbers but there are only 3 conversations (the last number is the end)
        self.lengths = [self.starts[i + 1] - self.starts[i] for i in range(len(self.starts) - 1)]

        # the order we go through the conversations: shuffled once with a fixed seed, so a resumed run finds the same order
        self.order = list(range(len(self.lengths))) # [0, 1, 2, ..., 403747]
        random.seed(1337)
        random.shuffle(self.order)                  # [3051, 52, 220871, 1033, ...]
        # Extra step, not in Andrej's code: put short conversations together and long ones together, so there's less padding.
        # Small example: one step of 8 conversations (instead of 32), micro-batches of 4. Their lengths after the shuffle:
        #   [900, 100, 800, 150, 120, 950, 130, 850]
        # without sorting:  [900, 100, 800, 150] -> every row padded to 900 -> 4 x 900 = 3600 computed, 1950 real
        #                   [120, 950, 130, 850] -> every row padded to 950 -> 4 x 950 = 3800 computed, 2050 real
        #                   -> 7400 computed for 4000 real tokens: 46% of the GPU work is wasted on padding
        # sorted:           [100, 120, 130, 150] -> padded to 150 -> 4 x 150 =  600 computed,  500 real
        #                   [800, 850, 900, 950] -> padded to 950 -> 4 x 950 = 3800 computed, 3500 real
        #                   -> 4400 computed for 4000 real tokens: only 9% wasted
        # The step still trains on the same 8 conversations; only which ones sit together in a micro-batch changes.
        # On the real data: 29% wasted without sorting, 6% with it -> training about 25% faster.
        #
        # Why sort only inside each step (group of 32) and not the whole dataset? Sorting everything would put all the
        # short conversations in the first steps and all the long ones in the last steps: the steps would not be random anymore.
        #
        # The line below: take the 32 numbers of one step, sort them by their conversation's length, put them back.
        for k in range(0, len(self.order), target_examples_per_step):
            self.order[k:k + target_examples_per_step] = sorted(self.order[k:k + target_examples_per_step], key=lambda i: self.lengths[i])
        self.reset()

    def reset(self):
        # state, start at the first conversation
        self.current_position = 0

    def next_batch(self):
        B = self.B
        # the next B conversation numbers, e.g. self.order[0:4], then self.order[4:8], ...
        indices = self.order[self.current_position : self.current_position + B]
        x, y = self.batch(indices)
        # advance position
        self.current_position += B
        # if loading the next batch would be out of bounds, reset (= start the next epoch)
        if self.current_position + B > len(self.order):
            self.reset()
        return x, y

    def batch(self, indices):
        """one row per conversation, right-padded to the longest one -> x, y (rows, row_len), y = -1 means no loss.
        The same as nanochat's collate function."""
        # indices = the conversations of this micro-batch (4 of them, with similar lengths: see the sorting in __init__).
        # They have different lengths, but a tensor must be a rectangle (every row the same length).
        # So: every row gets the length of the LONGEST conversation, and the empty space is filled with padding.
        #
        # indices are just conversation NUMBERS (their position in the dataset), not tokens. Where they come from:
        #   self.order = [3051, 52, 220871, 1033, ...]   all 403,748 conversation numbers, shuffled once (see __init__)
        #   next_batch() takes the next B = 4 numbers of self.order and calls batch() with them
        #
        # Example (2 conversations instead of 4, to keep it small):
        #   indices = [52, 1033]
        #   self.lengths[52]   = 7    (conversation 52   = self.ids[self.starts[52]   : self.starts[53]],   7 tokens)
        #   self.lengths[1033] = 9    (conversation 1033 = self.ids[self.starts[1033] : self.starts[1034]], 9 tokens)
        #   -> 2 rows (len(indices) = 2), and the longest has 9 tokens
        
        

        
        # Why -1? Every token's target is the token right after it (x = ids[:-1], y = ids[1:], same as pretraining):
        #   5 tokens  A B C D E   ->   x = A B C D
        #                              y = B C D E      (E is never an input: nothing comes after it to predict)
        # So a conversation of n tokens fills n-1 columns (even or odd n, always n-1).
        #
        # row_len = how long every row of x and y will be (longest conversation - 1).
        # Example, indices = [52, 1033]:  max(self.lengths[52], self.lengths[1033]) - 1  =  max(7, 9) - 1  =  8
        row_len =max(self.lengths[i] for i in indices) - 1

        # torch.full((rows, cols), value) makes a new tensor of that size and fills every cell with the same value.
        #   torch.zeros((2, 3))     -> [[0, 0, 0], [0, 0, 0]]        (always fills with 0)
        #   torch.full((2, 3), 7)   -> [[7, 7, 7], [7, 7, 7]]        (fills with any number you choose)
        # Here: x is filled with BOS and y with -1, both of size (len(indices), row_len) = (2, 8) in the example.
        # We start with a rectangle that is ALL padding, and the loop below writes each conversation over the start of its row.
        #
        #   x (2, 8):  BOS BOS BOS BOS BOS BOS BOS BOS        y (2, 8):  -1 -1 -1 -1 -1 -1 -1 -1
        #              BOS BOS BOS BOS BOS BOS BOS BOS                   -1 -1 -1 -1 -1 -1 -1 -1
        #
        # x is padded with BOS: x goes into the embedding table, so it needs a real token id (-1 would crash there).
        #   Which id doesn't matter: y is -1 at the same places, so the padding is never trained on.
        # y is padded with -1: -1 = "no loss here" (cross_entropy's ignore_index=-1).
        #
        # The padding sits at the END of each row (the right side): with causal attention a token only looks at the
        # tokens before it, so the real tokens never see the padding -> no attention mask needed.
        x = torch.full((len(indices), row_len), BOS, dtype=torch.long)
        y = torch.full((len(indices), row_len), -1, dtype=torch.long)
        for row, i in enumerate(indices):
            s, e = self.starts[i], self.starts[i + 1]
            ids = self.ids[s:e]
            mask = self.mask[s:e]
            n = e - s - 1
            x[row, :n] = ids[:-1]
            targets = ids[1:].clone() # clone = a copy, so setting -1 below doesn't change self.ids
            targets[mask[1:] == 0] = -1 # only assistant tokens (and <|assistant_end|>) are targets
            y[row, :n] = targets
        return x, y


@torch.no_grad()
def evaluate(model):
    # the same as the validation in pretraining: the same val batches every time, average of their losses
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y) # loss = mean over the assistant tokens (y = -1 is ignored, see model.py)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    model.train()
    return val_loss_accum.item()


def sample(model):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    max_new_tokens = 150
    for prompt in sample_prompts:
        ids = render_for_completion([{"role": "user", "content": prompt}])
        reply = generate_reply(model, ids, max_new_tokens=max_new_tokens, generator=generator)
        stopped = "" if len(reply) < max_new_tokens else f"  [didn't stop within {max_new_tokens} tokens]"
        log(f"Q: {prompt}\nA: {enc.decode(reply)}{stopped}")
    model.train()


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr * (max_steps - step) / (max_steps - warmup_steps) # linear decay to 0 at the last step


def save_checkpoint(path, step, val_loss, best_val_loss, with_optimizer):
    checkpoint = {
        'model': model.state_dict(),
        'config': asdict(model.config),
        'step': step,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_position': train_loader.current_position, # where we are in the data, like pretraining
    }
    if with_optimizer: # only ckpt_last needs it (to continue); ckpt_best stays small (~0.5GB instead of 1.5GB)
        checkpoint['optimizer'] = optimizer.state_dict()
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno()) # on the disk now, so a power cut can't leave a file of zeros behind
    os.replace(tmp_path, path)


train_loader = DataLoaderLite(B=B, split="train")
val_loader = DataLoaderLite(B=B, split="test")

steps_per_epoch = len(train_loader.order) // target_examples_per_step # 403,748 // 32 = 12,617
max_steps = steps_per_epoch * num_epochs


last_ckpt = os.path.join(log_dir, "ckpt_last.pt")
# continue automatically after a crash / power cut: if log/ckpt_last.pt exists, resume from it
# (forgetting to resume would restart from scratch and overwrite ckpt_best.pt). To start over: delete log/ckpt_last.pt
resume = os.path.exists(last_ckpt)
init_from = last_ckpt if resume else pretrained_ckpt
# no torch.compile here (unlike pretraining): every batch has a different length, so it would keep recompiling.
# Nothing wraps the model, so we don't need pretraining's raw_model: model is the plain GPT.
model, ckpt = load_model(init_from, device)
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device)

start_step, val_loss, best_val_loss = 0, None, float('inf')
if resume:
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step'] + 1
    val_loss, best_val_loss = ckpt['val_loss'], ckpt['best_val_loss']
    # put the train loader back where it stopped, so we don't re-train on the same conversations
    train_loader.current_position = ckpt['train_position']
# starting from pretraining: weights only, a fresh AdamW (the pretraining optimizer state is not loaded)
reset_metrics(keep_until_step=ckpt['step'] if resume else None)
del ckpt

log(f"{'resumed from' if resume else 'starting from'} {os.path.relpath(init_from, HERE)} at step {start_step}")
log(f"{len(train_loader.order):,} train conversations ({sum(train_loader.lengths):,} tokens), "
    f"{target_examples_per_step} per step -> {max_steps} steps, lr {max_lr:.0e}")
log(f"target examples per step: {target_examples_per_step}, B: {B} => grad accum steps: {grad_accum_steps}")

if start_step == 0:
    # baseline: the pretrained model in the chat format, before any SFT
    val_loss = evaluate(model)
    log(f"validation loss: {val_loss:.4f} (before SFT)")
    record(-1, val_loss=val_loss)
    sample(model)

for step in range(start_step, max_steps):
    last_step = step == max_steps - 1
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    tokens_processed = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y) # loss = mean over the assistant tokens (y = -1 is ignored, see model.py)
        # we have to scale the loss to account for gradient accumulation (same as pretraining)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        tokens_processed += x.numel() # B x row_len, including the little padding

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()

    dt = time.time() - t0
    tokens_per_sec = tokens_processed / dt
    log(f"step {step}/{max_steps}, loss: {loss_accum.item():.4f}, lr: {lr:.4e} | norm: {norm:.4f}, dt: {dt*1000:.0f}ms, tok/sec: {tokens_per_sec:.0f}")

    row = dict(train_loss=loss_accum.item(), lr=lr, norm=norm.item(), tok_per_sec=tokens_per_sec)
    if (step + 1) % eval_every == 0 or last_step:
        val_loss = evaluate(model)
        row["val_loss"] = val_loss
        log(f"validation loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(log_dir, "ckpt_best.pt"), step, val_loss, best_val_loss, with_optimizer=False)
    record(step, **row) # before ckpt_last is saved, so a resume never loses a row

    if (step + 1) % ckpt_every == 0 or last_step:
        t_save = time.time()
        save_checkpoint(last_ckpt, step, val_loss, best_val_loss, with_optimizer=True)
        log(f"saved ckpt_last.pt at step {step} ({time.time() - t_save:.1f}s)")

    if (step + 1) % sample_every == 0 or last_step:
        sample(model)

log(f"done. best val loss {best_val_loss:.4f} -> log/ckpt_best.pt, talk to it with: python chat.py")
