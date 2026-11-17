"""
Downloads HuggingFaceTB/smol-smoltalk and tokenizes it into the chat format (chat_format.py), once, before training.
Same idea as Andrej's fineweb.py: tokenize everything, put all the tokens in one long uint16 array, np.save it.
Writes to data/ (next to this file), per split:
    {split}_ids.npy      uint16, all conversations back to back (every conversation starts with BOS)
    {split}_mask.npy     uint8,  1 = train on this token
sft.py finds where each conversation starts by looking for BOS, like <|endoftext|> between documents in pretraining.
Run: python prepare_data.py
"""
import os
import multiprocessing as mp
from collections import Counter

import numpy as np
from datasets import load_dataset

from chat_format import ASSISTANT_END, render_conversation, visualize

MAX_LEN = 1025 # block_size 1024 + 1, because x = ids[:-1] and y = ids[1:]; longer conversations are cut at a turn boundary
DROP_SOURCES = {"self-oss-instruct"} # Python coding problems, FineWeb-EDU taught the model almost no code

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def tokenize(row):
    # one conversation -> (status, source, ids, mask). ids and mask are None if the conversation is dropped
    if row["source"] in DROP_SOURCES:
        return "dropped: code (self-oss-instruct)", row["source"], None, None
    try:
        ids, mask = render_conversation(row["messages"])
    except ValueError:
        return "dropped: bad role order", row["source"], None, None
    status = "kept"
    if len(ids) > MAX_LEN:
        # too long: keep the first turns, cut right after the last <|assistant_end|> that fits (a message boundary)
        ends = [i for i in range(MAX_LEN) if ids[i] == ASSISTANT_END]
        if not ends:
            return f"dropped: first reply longer than {MAX_LEN} tokens", row["source"], None, None
        ids, mask = ids[:ends[-1] + 1], mask[:ends[-1] + 1]
        status = "kept: cut to the turns that fit"
    # same as fineweb.py: uint16 = 2 bytes per token (our biggest id is 50260 < 2**16), uint8 = 1 byte per mask value
    ids_np = np.array(ids).astype(np.uint16)
    mask_np = np.array(mask).astype(np.uint8)
    return status, row["source"], ids_np, mask_np


def prepare(split):
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
    print(f"\n=== {split}: {len(ds):,} conversations ===")

    status_counts, kept_per_source = Counter(), Counter()
    conversations = [] # (ids, mask) of every kept conversation
    with mp.Pool(max(1, os.cpu_count() // 2)) as pool:
        for status, source, ids, mask in pool.imap(tokenize, ds, chunksize=256):
            status_counts[status] += 1
            if ids is not None:
                kept_per_source[source] += 1
                conversations.append((ids, mask))

    # one long array with all the conversations back to back, filled piece by piece like a fineweb.py shard
    total_tokens = sum(len(ids) for ids, mask in conversations)
    all_ids = np.empty((total_tokens,), dtype=np.uint16)
    all_mask = np.empty((total_tokens,), dtype=np.uint8)
    token_count = 0
    for ids, mask in conversations:
        all_ids[token_count:token_count + len(ids)] = ids
        all_mask[token_count:token_count + len(ids)] = mask
        token_count += len(ids)
    np.save(os.path.join(DATA_DIR, f"{split}_ids.npy"), all_ids)
    np.save(os.path.join(DATA_DIR, f"{split}_mask.npy"), all_mask)

    for status, n in sorted(status_counts.items()):
        print(f"{status:40s} {n:,}")
    print("kept per source:", dict(kept_per_source.most_common()))
    trained_tokens = int(all_mask.sum())
    print(f"total tokens: {total_tokens:,}   trained tokens (mask=1): {trained_tokens:,} ({trained_tokens / total_tokens:.0%})")
    lengths = sorted(len(ids) for ids, mask in conversations)
    print(f"length per conversation: mean {total_tokens / len(lengths):.0f}, median {lengths[len(lengths) // 2]}, "
          f"90% {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    return conversations


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    prepare("test")
    conversations = prepare("train")
    # Step 2 check: only the assistant's text (and <|assistant_end|>) should be inside [...]
    print("\n=== example (trained parts in [...]) ===")
    ids, mask = conversations[0]
    print(visualize(ids.tolist(), mask.tolist()))
