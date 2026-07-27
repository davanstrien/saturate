"""Tier 1 Job A — LightOnOCR-2 on vLLM through the CLEAN pumpjack package.

The pilot shape: real engine boot template, real vllm gauges, Auto controller,
hf://datasets output with manifest resume. Run twice with a deliberate cancel
in between to prove cross-job kill/resume over remote fs (never tested before).

On the Job (vllm/vllm-openai image, system python3, pumpjack wheel installed):
    python3 tier1_ocr.py --limit 100
"""

from __future__ import annotations

import argparse
import base64
import io

from pumpjack import Auto, Engine, pump

MODEL = "lightonai/LightOnOCR-2-1B"
INPUT = "davanstrien/moh-bench-sample"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-moh/data"
MAX_SIDE = 1540  # LightOnOCR-2 trained at 1540px max resolution


def image_to_data_uri(img) -> str:
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-moh", repo_type="dataset",
                        private=True, exist_ok=True)
    ds = load_dataset(INPUT, split="train")
    ds = ds.select(range(min(args.limit, len(ds))))
    rows = [(f"{ex['b_number']}-p{ex['page_index']:04d}", {"uri": image_to_data_uri(ex["image"])})
            for ex in ds]
    print(f"[tier1] loaded {len(rows)} pages", flush=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": row["uri"]}}]}],
                "max_tokens": args.max_tokens, "temperature": 0.2, "top_p": 0.9}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"markdown": body["choices"][0]["message"]["content"], "model": MODEL,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    with Engine(MODEL, engine="vllm",
                extra_args=["--max-model-len", "16384", "--gpu-memory-utilization", "0.90",
                            "--no-enable-prefix-caching"]) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     window=Auto(target_waiting=8, initial=8))
    print("TIER1_OCR " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
