"""Embeddings at scale — FOUR jobs, 200k fineweb-edu texts through vLLM pooling.

Scale-up of the Jul 27 tier1_embed spike (32k texts, 33.5k tok/s, window shrank
to 2 against a server that served one 64-seq batch at a time). This run adds
`--max-num-seqs 256` to the engine to test whether the ceiling the controller
discovered was a server-config artefact rather than a hard pooling limit.

On each Job (vllm/vllm-openai image, pumpjack wheel installed):
    python3 embed_4job.py --rank R --world 4
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, pump
from saturate.source import shard_select

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REPO = "davanstrien/pumpjack-embed-4job"
OUTPUT = f"hf://datasets/{REPO}/data"
BATCH = 64
N_BATCHES = 3125  # 3,125 x 64 = 200,000 texts


def batch_rows():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf, b = [], 0
    for ex in ds:
        text = (ex.get("text") or "")[:1200]
        if len(text) > 100:
            buf.append(text)
        if len(buf) == BATCH:
            yield (f"batch-{b:05d}", {"texts": buf})
            buf, b = [], b + 1
        if b >= N_BATCHES:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    HfApi().create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL, "input": row["texts"]}

    def parse(row: dict, body: dict) -> dict:
        data = body["data"]
        if len(data) != len(row["texts"]):  # schema-invalid: never mark as success
            raise ValueError(f"expected {len(row['texts'])} embeddings, got {len(data)}")
        usage = body.get("usage") or {}
        return {"texts": row["texts"],
                "embeddings": [d["embedding"] for d in data],
                "n_texts": len(data),
                "prompt_tokens": usage.get("prompt_tokens")}

    rows = shard_select(batch_rows(), rank=args.rank, world=args.world)
    with Engine(MODEL, engine="vllm",
                extra_args=["--runner", "pooling", "--convert", "embed",
                            "--max-num-seqs", "256",
                            "--gpu-memory-utilization", "0.90"]) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     route="/embeddings",
                     window=Auto(target_waiting=8, initial=4, max_limit=64),
                     shard=(args.rank, args.world), flush_every=50)
    texts_s = round(stats.rows_processed * BATCH / stats.elapsed_s, 1) if stats.elapsed_s else 0
    print(f"EMBED_SCALE rank={args.rank} texts_per_sec={texts_s} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
