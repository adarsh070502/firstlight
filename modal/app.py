"""
Firstlight Chat: Gradio app with the two Firstlight-124M models (deployed on Modal by modal_app.py).
    Chat tab:                       talk to the SFT model
    Before vs after training tab:   the same question to the base and the SFT model, side by side

Run locally:  python GPT2/modal/app.py   -> http://127.0.0.1:7865  (downloads the weights from Hugging Face)
"""
import gradio as gr
import torch

from model import load_model, render_chat, stream, enc

device = "cuda" if torch.cuda.is_available() else "cpu"   # Modal runs it on CPU
models = {name: load_model(name, device) for name in ("base", "chat")}

INTRO = """
# Firstlight-124M: a ChatGPT-style language model I built and trained from scratch

I built a **124-million-parameter language model** (GPT-2 architecture) in PyTorch and trained it from random
weights, in two stages:

1. **Pretraining:** it read **9.8 billion tokens** (about 7 billion words) of educational web text over ~7 days,
   learning English and general knowledge by predicting the next word. On the **HellaSwag** common-sense benchmark it
   scores **30.3%**, above OpenAI's original GPT-2 of the same size (29.6%).
2. **Fine-tuning into an assistant (SFT):** I then trained it on **400K example conversations**, with the loss only
   on the assistant's replies, so it answers a question and stops instead of just continuing the text.

👉 **Try it:** click an example question in the chat or type your own. The **"Before vs after training"** tab asks
the same question to the model after stage 1 and after stage 2, so you can see what fine-tuning changed.

**Keep in mind:** it's thousands of times smaller than ChatGPT, so its answers are fluent and well-structured but the
**facts are often wrong**. It remembers up to 1,024 tokens of the conversation. Served on Modal (CPU, with a KV cache).

**Model card and numbers:** [adarsh-007/firstlight-124m](https://huggingface.co/adarsh-007/firstlight-124m) ·
**Code:** [github.com/adarsh070502/firstlight](https://github.com/adarsh070502/firstlight)
"""

EXAMPLES = [
    "What is photosynthesis?",
    "Hi! Can you give me three tips for studying for an exam?",
    "Write a short poem about the ocean.",
    "Explain what a black hole is to a 10 year old.",
    "What are the benefits of regular exercise?",
]


def text_of(content):
    # Gradio gives a message's content as a string, or as a list of parts ({"type": "text", "text": ...})
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def chat_fn(message, history, temperature, top_k, max_new_tokens):
    messages = [{"role": m["role"], "content": text_of(m["content"])} for m in history if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": message})
    ids = render_chat(messages)
    # the model sees 1024 tokens: forget the oldest turns until the conversation + the reply fit
    while len(ids) + max_new_tokens > 1024 and len(messages) > 1:
        messages = messages[2:]
        ids = render_chat(messages)
    reply = ""
    for reply in stream(models["chat"], ids, int(max_new_tokens), temperature, int(top_k), chat=True):
        yield reply
    if not reply:
        yield "(empty reply)"


def compare_fn(prompt, temperature, top_k, max_new_tokens):
    if not prompt.strip():
        yield "", ""   # an empty prompt would give the model 0 tokens
        return
    # base: the prompt as plain text, it continues it. chat: the prompt as a user message, it answers
    base_ids = enc.encode_ordinary(prompt.rstrip())   # a trailing space becomes its own token and hurts predictions
    chat_ids = render_chat([{"role": "user", "content": prompt}])
    base_out = chat_out = ""
    for base_out in stream(models["base"], base_ids, int(max_new_tokens), temperature, int(top_k), chat=False):
        yield base_out, chat_out
    for chat_out in stream(models["chat"], chat_ids, int(max_new_tokens), temperature, int(top_k), chat=True):
        yield base_out, chat_out


with gr.Blocks(title="Firstlight: an AI built from scratch") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature (0 = always the most likely word)")
        top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k (choose among the k most likely words)")
        max_new_tokens = gr.Slider(20, 500, value=300, step=20, label="Max answer length (tokens)")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            chat_fn,
            additional_inputs=[temperature, top_k, max_new_tokens],
            examples=[[e] for e in EXAMPLES],
            cache_examples=False,
        )

    with gr.Tab("Before vs after training"):
        gr.Markdown("The **same question** to the model **before** and **after** I trained it to be an assistant. "
                    "Before: it only knows how to continue text, so it rambles. After: it answers the question.")
        prompt = gr.Textbox(label="Question", value=EXAMPLES[1], lines=2)
        run = gr.Button("Ask both", variant="primary")
        with gr.Row():
            base_box = gr.Textbox(label="Before: it only learned to read English", lines=14)
            chat_box = gr.Textbox(label="After: trained to be an assistant", lines=14)
        gr.Examples(EXAMPLES, inputs=prompt)
        run.click(compare_fn, [prompt, temperature, top_k, max_new_tokens], [base_box, chat_box])
        prompt.submit(compare_fn, [prompt, temperature, top_k, max_new_tokens], [base_box, chat_box])

if __name__ == "__main__":
    demo.launch(server_port=7865)
