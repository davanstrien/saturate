"""TEI Job-to-Job — CPU client Job pumps an exposed TEI GPU Job over the proxy.

The production fan-out arrangement for embeddings: serving and pumping are
separate Jobs; the client is blind (TEI metrics port not proxied) and cheap
(cpu flavor). Same OpenAI /v1/embeddings route, zero library changes.

On the client Job:  python3 tei_jobs.py <exposed-url>
"""

from __future__ import annotations

import sys

from huggingface_hub import get_token

from saturate import Auto, pump, wait_for_health

BATCH = 32
N_BATCHES = 200
OUTPUT = "hf://datasets/davanstrien/pumpjack-tier1-tei/data"


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
    from huggingface_hub import HfApi

    HfApi().create_repo("davanstrien/pumpjack-tier1-tei", repo_type="dataset",
                        private=True, exist_ok=True)
    base = sys.argv[1].rstrip("/")
    endpoint = f"{base}/v1"
    headers = {"Authorization": f"Bearer {get_token()}"}

    def to_request(row: dict) -> dict:
        return {"model": "tei", "input": row["texts"]}

    def parse(row: dict, body: dict) -> dict:
        data = body["data"]
        if len(data) != len(row["texts"]):
            raise ValueError(f"expected {len(row['texts'])} embeddings, got {len(data)}")
        return {"texts": row["texts"],
                "embeddings": [d["embedding"] for d in data],
                "n_texts": len(data),
                "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}

    wait_for_health(endpoint, 900, headers=headers)
    stats = pump(batch_rows(), to_request, parse, endpoint, OUTPUT,
                 route="/embeddings", headers=headers,
                 window=Auto(target_waiting=8, initial=4, max_limit=32),
                 flush_every=25)
    print("TEI_JOBS " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
