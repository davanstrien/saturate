"""Tier 1 fan-out — K shards, K engines, ONE output directory.

The scaling story (fan-out-to-storage, never a live cluster): each Job hosts
its own engine and pumps its strided slice of the global stream into a shared
output. Proves shard_select, disjoint id spaces, concurrent writers, and the
per-shard marker space on the clean build.

On each Job:  python3 tier1_fanout.py --rank R --world 4
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, pump
from saturate.source import shard_select

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-fanout/data"
N = 4000


def global_rows():
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    n = 0
    for ex in ds:
        if ex["instruction"]:
            yield (f"dolly-{n:05d}", {"text": ex["instruction"][:2000]})
            n += 1
        if n >= N:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-fanout", repo_type="dataset",
                        private=True, exist_ok=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL, "messages": [{"role": "user", "content": row["text"]}],
                "max_tokens": 150, "temperature": 0.7}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"out": body["choices"][0]["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    rows = shard_select(global_rows(), rank=args.rank, world=args.world)
    with Engine(MODEL, engine="vllm",
                extra_args=["--gpu-memory-utilization", "0.90",
                            "--no-enable-prefix-caching"]) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     window=Auto(target_waiting=8, initial=8),
                     shard=(args.rank, args.world), flush_every=50)
    print(f"TIER1_FANOUT rank={args.rank} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
