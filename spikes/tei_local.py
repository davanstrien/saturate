"""TEI backend test — laptop pump vs an exposed TEI Job. BLIND MODE, live.

TEI serves the OpenAI-compatible /v1/embeddings route; its Prometheus metrics
live on a port the jobs proxy doesn't carry, so this is the first real-world
run of mode C: remote endpoint, no gauges, AIMD floor only. Also the first
run through the exposed-Job proxy with auth headers.

Run locally:  uv run --with datasets python spikes/tei_local.py <exposed-url>
"""

from __future__ import annotations

import sys

from huggingface_hub import get_token

from saturate import Auto, pump, wait_for_health

BATCH = 32
N_BATCHES = 100
OUTPUT = "/tmp/pumpjack-tei-test"


def batch_rows():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf, b = [], 0
    for ex in ds:
        text = (ex.get("text") or "")[:800]
        if len(text) > 100:
            buf.append(text)
        if len(buf) == BATCH:
            yield (f"batch-{b:05d}", {"texts": buf})
            buf, b = [], b + 1
        if b >= N_BATCHES:
            return


def main() -> None:
    base = sys.argv[1].rstrip("/")
    endpoint = f"{base}/v1"
    headers = {"Authorization": f"Bearer {get_token()}"}

    def to_request(row: dict) -> dict:
        return {"model": "tei", "input": row["texts"]}

    def parse(row: dict, body: dict) -> dict:
        data = body["data"]
        if len(data) != len(row["texts"]):
            raise ValueError(f"expected {len(row['texts'])} embeddings, got {len(data)}")
        return {"n_texts": len(data), "dim": len(data[0]["embedding"]),
                "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}

    wait_for_health(endpoint, 900, headers=headers)
    stats = pump(batch_rows(), to_request, parse, endpoint, OUTPUT,
                 route="/embeddings", headers=headers,
                 window=Auto(target_waiting=8, initial=4, max_limit=32),
                 flush_every=25)
    print("TEI_LOCAL " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
