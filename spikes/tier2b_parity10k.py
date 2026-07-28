"""Parity A/B at fair scale: 10k rows per arm, SAME rows, prefix caching off.

The 2k-row run scored 0.515 because Auto's discovery ramp (8 -> ~128 under the
ACK-clock) is a fixed cost that dominates a 44s run. At 10k rows the ramp is
~5% of the run; this is the honest M1 parity test (bar: >= 0.95 of bare).
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from pumpjack import Auto, Engine, pump

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N = 10_000
MAX_TOKENS = 150


def dolly_rows(n: int):
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    got = 0
    for i, ex in enumerate(ds):
        if not ex["instruction"]:
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


async def bare_arm(endpoint: str) -> dict:
    sem = asyncio.Semaphore(64)
    tokens = 0
    rows = list(dolly_rows(N))
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
        bare = asyncio.run(bare_arm(endpoint))
        print("PARITY10K_BARE " + json.dumps(bare), flush=True)
        s = pump(dolly_rows(N), to_request, parse, endpoint, "/tmp/parity10k",
                 window=Auto(target_waiting=8, initial=8), flush_every=500)
        print("PARITY10K_PUMP " + s.to_json(), flush=True)
        ratio = s.tokens_per_sec / bare["tok_s"] if bare["tok_s"] else 0
        print(f"PARITY10K_RATIO {ratio:.3f} (bar: >=0.95)", flush=True)


if __name__ == "__main__":
    main()
