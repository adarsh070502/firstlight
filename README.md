# Firstlight-124M

A 124M-parameter language model **pretrained from scratch** on 9.83B tokens of FineWeb-EDU, then turned into a
**chat assistant** with supervised fine-tuning (SFT). GPT-2 architecture, plain PyTorch.

- **Demo** (Base vs Chat side by side): https://adarsheluri143--firstlight-chat-ui.us-east.modal.run
- **Weights + model card:** https://huggingface.co/adarsh-007/firstlight-124m

| Model | HellaSwag (acc_norm) | Val loss |
|---|---|---|
| OpenAI GPT-2 124M | 29.55% | - |
| **Firstlight base** (pretrained) | **30.30%** | 3.0436 (FineWeb-EDU) |
| **Firstlight chat** (after SFT) | 29.81% | 1.4495 (assistant tokens, smol-smoltalk test) |

```
Q: Hi! Can you give me three tips for studying for an exam?

base: =/?src=?src=?src=?src=...                        (in the chat format: never stops)
chat: 1. **Evaluate your answers**: Make a note of your answers to the exam questions. ...
      2. **Take your time**: Make a plan for studying. ...
      3. **Use study materials**: Choose study materials that are relevant to your studies ...
```

## The two stages

| Stage | Folder | What happens |
|---|---|---|
| 1. Pretraining | [`pretraining/`](pretraining) | GPT-2 124M from random weights on FineWeb-EDU `sample-10BT`: 18,751 steps × 0.5M tokens (micro-batch 4 × 128 gradient accumulation), AdamW, warmup + cosine LR, bf16, `torch.compile`, flash attention, checkpoint/resume every 100 steps. ~7 days. |
| 2. SFT | [`sft/`](sft) | 403,748 smol-smoltalk conversations with 4 chat tokens in the vocab's padded slots; the loss only on the assistant's tokens and `<\|assistant_end\|>`. 1 epoch, 12,617 steps × 32 conversations, LR 5e-5, 5.8 h. |

## Files

```
pretraining/
  fineweb.py        download + tokenize FineWeb-EDU sample-10BT into 100M-token shards
  gpt2.py           the model and the pretraining loop (resumes from log/ckpt_last.pt)
  hellaswag.py      HellaSwag eval (own checkpoint or any HF GPT-2), logs every question + answer
  inference.py      text completion with the base model
  log/log.txt       the full pretraining log
sft/
  model.py          the same GPT, cross-entropy with ignore_index=-1
  chat_format.py    special tokens, conversation -> (ids, loss mask), decoding, generation
  prepare_data.py   download + tokenize smol-smoltalk (filtering, cutting long chats at a turn boundary)
  sft.py            the SFT loop: one conversation per row, padding sorted per step, auto-resume
  chat.py           talk to the model in the terminal
  hellaswag.py      did SFT keep the pretraining knowledge?
  log/              log.txt + metrics.csv of the SFT run
```

Checkpoints, datasets and tokenized data are not in this repo (too large). The weights are on
[Hugging Face](https://huggingface.co/adarsh-007/firstlight-124m); the data is rebuilt by `fineweb.py` and
`prepare_data.py`.

## Run it

```bash
pip install torch tiktoken numpy datasets

# reproduce the training (scripts expect to be run from the folder that contains this repo as GPT2/)
python pretraining/fineweb.py                    # ~10B tokens of shards
python pretraining/gpt2.py                       # pretraining
python sft/prepare_data.py && python sft/sft.py  # SFT
python sft/chat.py                               # chat in the terminal
```

## Credits and license

- Andrej Karpathy
- Data: FineWeb-EDU (ODC-By 1.0) and smol-smoltalk (Apache-2.0), both by Hugging Face.
- Code and weights: MIT.
