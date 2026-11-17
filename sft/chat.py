"""
Talk to the SFT model.
    python chat.py                          interactive (/reset clears the conversation, /quit exits)
    python chat.py --prompt "What is DNA?"  one question, one answer
    python chat.py --ckpt ../pretraining/log/ckpt_best.pt   the base model, to compare
"""
import argparse
import os

import torch

from chat_format import enc, generate_reply, render_for_completion
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.path.join(HERE, "log", "ckpt_best.pt"))
parser.add_argument("--prompt", default=None, help="ask one question and exit")
parser.add_argument("--temperature", type=float, default=0.7) # 0 = always the most likely token
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=300)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = load_model(args.ckpt, device)
model.eval()
print(f"loaded {args.ckpt} (step {ckpt.get('step')}, val loss {ckpt.get('val_loss'):.4f})")
del ckpt


def answer(messages):
    ids = render_for_completion(messages)
    # the model only sees 1024 tokens: forget the oldest turns until the conversation + the reply fit
    while len(ids) + args.max_new_tokens > model.config.block_size and len(messages) > 1:
        messages = messages[2:]
        ids = render_for_completion(messages)
    reply = generate_reply(model, ids, args.max_new_tokens, args.temperature, args.top_k)
    return messages, enc.decode(reply)


if args.prompt is not None:
    _, reply = answer([{"role": "user", "content": args.prompt}])
    print(reply)
else:
    messages = []
    print("type /reset to start over, /quit to exit")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            messages = []
            print("(conversation cleared)")
            continue
        messages, reply = answer(messages + [{"role": "user", "content": text}])
        print(f"Assistant: {reply}")
        messages = messages + [{"role": "assistant", "content": reply}]
