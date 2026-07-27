"""Tier 1 big-OCR soak — 5k London's Pulse MOH pages, TRUE streaming vision.

The base64/data-URI conversion happens lazily inside the generator, so RAM
stays flat (the POC needed hand-rolled 10x1000 chunking for exactly this).
CPU-side image encoding racing the GPU is a live test of input-bound
detection — if the source can't keep up, the freeze + hint should fire.

On the Job (vllm/vllm-openai image, pumpjack wheel installed):
    python3 tier1_bigocr.py
"""

from __future__ import annotations

import base64
import io

from pumpjack import Auto, Engine, pump

MODEL = "lightonai/LightOnOCR-2-1B"
INPUT = "biglam/londons-pulse-moh"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-moh5k/data"
MAX_SIDE = 1540
N = 5000


def stream_rows():
    from datasets import load_dataset

    ds = load_dataset(INPUT, split="train", streaming=True)
    n = 0
    for ex in ds:
        img = ex["image"].convert("RGB")
        if max(img.size) > MAX_SIDE:
            scale = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        yield (f"{ex['b_number']}-p{ex['page_index']:04d}", {"uri": uri})
        n += 1
        if n >= N:
            return


def main() -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-moh5k", repo_type="dataset",
                        private=True, exist_ok=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": row["uri"]}}]}],
                "max_tokens": 4096, "temperature": 0.2, "top_p": 0.9}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"markdown": body["choices"][0]["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    with Engine(MODEL, engine="vllm",
                extra_args=["--max-model-len", "16384", "--gpu-memory-utilization", "0.90",
                            "--no-enable-prefix-caching"]) as endpoint:
        stats = pump(stream_rows(), to_request, parse, endpoint, OUTPUT,
                     window=Auto(target_waiting=8, initial=8), flush_every=50)
    print("TIER1_BIGOCR " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
