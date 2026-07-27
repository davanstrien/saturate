"""Tier 1 embeddings — /v1/embeddings route + micro-batch-as-row, ZERO library changes.

The decision-13 thesis test: if pumpjack is genuinely OpenAI-HTTP-generic, an
embedding workload needs only route="/embeddings" and a row that is a
pre-grouped batch (decision 12's seam: batch-id as id, array columns out).
32k fineweb-edu texts in 64-text batches = 500 rows. Baseline to compare:
the Jul 9 datatrove fleet did ~615 docs/s per L4.

On the Job (vllm/vllm-openai image, pumpjack wheel installed):
    python3 tier1_embed.py
"""

from __future__ import annotations

from pumpjack import Auto, Engine, pump

MODEL = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-embed/data"
BATCH = 64
N_BATCHES = 500


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
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-embed", repo_type="dataset",
                        private=True, exist_ok=True)

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

    with Engine(MODEL, engine="vllm",
                extra_args=["--runner", "pooling", "--convert", "embed",
                            "--gpu-memory-utilization", "0.90"]) as endpoint:
        stats = pump(batch_rows(), to_request, parse, endpoint, OUTPUT,
                     route="/embeddings",
                     window=Auto(target_waiting=8, initial=4, max_limit=64),
                     flush_every=25)
    print("TIER1_EMBED " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
