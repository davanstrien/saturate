"""Tier 1 Job C — the synthetic-data shape: streaming dataset in, text gen out.

Exercises what A and B don't: a TRUE streaming source (generator over a
streaming HF dataset — never materialized, the EBDC lesson) and content-hash
id derivation (dict rows, no id column — the CONTRACT §2 default, first live
run). 5k rows; if the streaming source can't keep up, the input-bound freeze
and its advisor hint should fire — that's a feature under test, not a failure.

On the Job (vllm/vllm-openai image, system python3, pumpjack wheel installed):
    python3 tier1_synth.py
"""

from __future__ import annotations

from pumpjack import Auto, Engine, pump

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-synth/data"
N = 5000


def stream_rows():
    """Generator over the streaming dataset — dict rows, NO ids: pumpjack
    derives content-hash ids (first live run of the default id path)."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    n = 0
    for ex in ds:
        if ex["instruction"]:
            yield {"instruction": ex["instruction"][:2000], "category": ex["category"]}
            n += 1
        if n >= N:
            return


def main() -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-synth", repo_type="dataset",
                        private=True, exist_ok=True)

    def to_request(row: dict) -> dict:
        return {"model": MODEL,
                "messages": [{"role": "user", "content": row["instruction"]}],
                "max_tokens": 150, "temperature": 0.7}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"instruction": row["instruction"], "category": row["category"],
                "response": body["choices"][0]["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    with Engine(MODEL, engine="vllm",
                extra_args=["--gpu-memory-utilization", "0.90",
                            "--no-enable-prefix-caching"]) as endpoint:
        stats = pump(stream_rows(), to_request, parse, endpoint, OUTPUT,
                     window=Auto(target_waiting=8, initial=8), flush_every=100)
    print("TIER1_SYNTH " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
