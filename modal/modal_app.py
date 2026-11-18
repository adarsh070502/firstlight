"""
Firstlight Chat on Modal: the Gradio app (app.py + model.py, full fp32 model, server-side, KV cache), served as a web endpoint.

    modal serve GPT2/modal/modal_app.py     temporary URL, reloads on edits (testing)
    modal deploy GPT2/modal/modal_app.py    permanent URL: https://adarsheluri143--firstlight-chat-ui.us-east.modal.run

Run the app locally instead:  python GPT2/modal/app.py   -> http://127.0.0.1:7865

Cost: it runs only while someone uses it and shuts down 2 min after the last request (billed per second of use;
free credits cover a demo). The first visit after a shutdown waits for a cold start (~20-30 s: start the container, load the
two models); the weights are baked into the image, so nothing is downloaded at that point.
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
WEIGHTS = ["config.json", "base/model.safetensors", "chat/model.safetensors"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gradio==6.2.0", "fastapi", "tiktoken", "safetensors", "huggingface_hub")
    .env({"HF_HOME": "/root/hf", "TIKTOKEN_CACHE_DIR": "/root/tiktoken"})
    # download the weights (and tiktoken's GPT-2 vocab) while building the image, not on every cold start
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"[d('adarsh-007/firstlight-124m', f) for f in {WEIGHTS}]\"",
        "python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"",
    )
    .add_local_file(HERE / "app.py", "/root/app/app.py")
    .add_local_file(HERE / "model.py", "/root/app/model.py")
)

app = modal.App("firstlight-chat", image=image)


@app.function(cpu=2, memory=4096, scaledown_window=120, max_containers=1)   # Gradio needs one container (sticky sessions)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    import sys
    sys.path.insert(0, "/root/app")
    import gradio as gr
    from fastapi import FastAPI
    from app import demo   # loads both models (downloads skipped: they're in the image's HF cache)

    return gr.mount_gradio_app(FastAPI(), demo, path="/")
