"""Tier 1 Job B — SGLang boot template + LIVE dialect validation.

Current SGLang emits the `sglang_` prefix (renamed from `sglang:` in v0.5.4);
this run proves the dual-prefix regex against the real thing, plus the sglang
boot template and the controller on a text workload.

On the Job (lmsysorg/sglang image, system python3, pumpjack wheel installed):
    python3 tier1_sglang.py
"""

from __future__ import annotations

from saturate import Auto, Engine, pump

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-sglang/data"
N = 400


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-sglang", repo_type="dataset",
                        private=True, exist_ok=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if ex["instruction"]:
            rows.append((f"dolly-{i:05d}", {"text": ex["instruction"][:2000]}))
        if len(rows) >= N:
            break
    print(f"[tier1] loaded {len(rows)} prompts", flush=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL,
                "messages": [{"role": "user", "content": row["text"]}],
                "max_tokens": 200, "temperature": 0.0}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"out": body["choices"][0]["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    with Engine(MODEL, engine="sglang",
                extra_args=["--mem-fraction-static", "0.85"]) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     window=Auto(target_waiting=8, initial=8))
    print("TIER1_SGLANG " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
