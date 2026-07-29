"""Inference Endpoints — the fourth serving arrangement (managed, scale-to-zero).

Dedicated IE endpoint: Qwen2.5-0.5B-Instruct on the vllm-openai custom image,
L4 x1, min_replica=0, scale_to_zero_timeout=15. IE proxies no engine metrics,
so both arms run BLIND (mode C: AIMD floor only).

  warm: wait_for_health, then 1k dolly rows — window/throughput receipt.
  cold: pump straight into a scaled-to-zero endpoint WITHOUT wait_for_health —
        the receipt is the retry ladder + breaker riding the managed-wake
        responses until the replica arrives; rows must heal, zero lost.

Run locally:
    uv run --with datasets python spikes/endpoints_qwen.py warm <endpoint-url>
    hf endpoints scale-to-zero saturate-ie-spike   # then, once scaledToZero:
    uv run --with datasets python spikes/endpoints_qwen.py cold <endpoint-url>
"""

from __future__ import annotations

import sys

from huggingface_hub import get_token

from saturate import Auto, pump, wait_for_health

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N = {"warm": 1000, "cold": 100, "scale": 6000}  # scale: long enough for replica 2 to arrive


def rows(n: int):
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    got = 0
    for ex in ds:
        if ex["instruction"]:
            yield {"instruction": ex["instruction"][:2000], "category": ex["category"]}
            got += 1
        if got >= n:
            return


def main() -> None:
    arm = sys.argv[1]
    base = sys.argv[2].rstrip("/")
    endpoint = f"{base}/v1"
    headers = {"Authorization": f"Bearer {get_token()}"}
    output = f"/tmp/pumpjack-ie-{arm}"

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

    if arm == "warm":
        wait_for_health(endpoint, 900, headers=headers)
    stats = pump(rows(N[arm]), to_request, parse, endpoint, output,
                 headers=headers, window=Auto(initial=8), flush_every=100)
    print(f"IE_{arm.upper()} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
