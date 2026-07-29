"""Tier 2 pair in one job: bucket-sink validation + bare-httpx parity A/B.

Arm 1 — bucket sink: 1,000 dolly rows -> hf://buckets/... (the staged-output
prerequisite: ParquetSink over a bucket URI has never been validated).
Then kill nothing but re-run pump() in-process against the same bucket output
to verify resume reads the bucket manifest (rows_done_prior == 1000).

Arm 2 — parity: same server, 2,000 FRESH dolly rows. bare = well-tuned
plain-httpx client (fixed semaphore 64, no retries beyond one, minimal parse)
vs pump(Auto). Bar: pump tok/s >= 95% of bare (M1 acceptance).
Prefix caching disabled; distinct row ranges per arm (no cache pollution).

On the Job (vllm/vllm-openai image, pumpjack wheel installed):
    python3 tier2_bucket_parity.py
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from saturate import Auto, Engine, pump

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BUCKET_OUT = "hf://buckets/davanstrien/pumpjack-scratch/tier2-synth/data"
MAX_TOKENS = 150


def dolly_rows(skip: int, n: int):
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    got = 0
    for i, ex in enumerate(ds):
        if i < skip or not ex["instruction"]:
            continue
        yield (f"dolly-{i:05d}", {"text": ex["instruction"][:2000]})
        got += 1
        if got >= n:
            return


def to_request(row: dict) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": row["text"]}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0}


def parse(row: dict, body: dict) -> dict:
    usage = body.get("usage") or {}
    return {"out": body["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


async def bare_arm(endpoint: str, rows: list) -> dict:
    """The hand-tuned baseline an expert would write: fixed 64, one retry."""
    sem = asyncio.Semaphore(64)
    tokens = 0
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(30, read=600)) as client:

        async def one(row):
            nonlocal tokens
            async with sem:
                for _ in range(2):
                    try:
                        r = await client.post(f"{endpoint}/chat/completions", json=to_request(row))
                        if r.status_code == 200:
                            u = r.json().get("usage") or {}
                            tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                            return
                    except httpx.HTTPError:
                        pass

        await asyncio.gather(*[one(row) for _, row in rows])
    dt = time.monotonic() - t0
    return {"arm": "bare-httpx-64", "wall_s": round(dt, 1), "tokens": tokens,
            "tok_s": round(tokens / dt, 1), "n": len(rows)}


def main() -> None:
    with Engine(MODEL, engine="vllm",
                extra_args=["--gpu-memory-utilization", "0.90",
                            "--no-enable-prefix-caching"]) as endpoint:
        # --- Arm 1: bucket sink + bucket resume ---
        s1 = pump(dolly_rows(0, 1000), to_request, parse, endpoint, BUCKET_OUT,
                  window=Auto(target_waiting=8, initial=8), flush_every=100)
        print("BUCKET_RUN " + s1.to_json(), flush=True)
        s2 = pump(dolly_rows(0, 1000), to_request, parse, endpoint, BUCKET_OUT,
                  window=Auto(target_waiting=8, initial=8), flush_every=100)
        print("BUCKET_RESUME " + s2.to_json(), flush=True)  # want rows_done_prior=1000

        # --- Arm 2: parity, fresh rows per sub-arm ---
        bare = asyncio.run(bare_arm(endpoint, list(dolly_rows(2000, 2000))))
        print("PARITY_BARE " + json.dumps(bare), flush=True)
        s3 = pump(dolly_rows(5000, 2000), to_request, parse, endpoint,
                  "/tmp/parity-pump", window=Auto(target_waiting=8, initial=8),
                  flush_every=200)
        print("PARITY_PUMP " + s3.to_json(), flush=True)
        ratio = s3.tokens_per_sec / bare["tok_s"] if bare["tok_s"] else 0
        print(f"PARITY_RATIO {ratio:.3f} (bar: >=0.95)", flush=True)


if __name__ == "__main__":
    main()
