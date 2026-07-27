"""AsyncLLM spike (2026-07-27): does naive in-process feeding keep the GPU fed?

Three arms, same model/prompts/sampling, each in a fresh subprocess (clean VRAM):
  offline   — vllm.LLM.generate on the full list (the classic offline batch)
  naive     — AsyncLLM, one asyncio task per prompt, all submitted at once
  windowed  — AsyncLLM, bounded feed-ahead window (2x max_num_seqs)

Measures wall seconds, completion tokens/sec, peak RSS. Run on a GPU Job with
the vllm/vllm-openai image (system python — vllm preinstalled).

Questions this answers (thread receipt for Harry's AsyncLLM steer):
  1. Does naive submission match windowed/offline throughput? (engine scheduler
     should batch identically — the client just holds more state)
  2. What does naive submission cost in RSS at 2k prompts? (extrapolates to the
     1M-row / multimodal case where it matters)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 2000
MAX_TOKENS = 200
MAX_NUM_SEQS = 256
ARMS = ("offline", "naive", "windowed")


def load_prompts() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    prompts = []
    for row in ds:
        if row["instruction"]:
            prompts.append(row["instruction"][:2000])
        if len(prompts) >= N_PROMPTS:
            break
    return prompts


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)  # linux: KB


def sampling():
    from vllm import SamplingParams

    return SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True)


def arm_offline(prompts: list[str]) -> dict:
    from vllm import LLM

    llm = LLM(model=MODEL, max_num_seqs=MAX_NUM_SEQS, gpu_memory_utilization=0.90)
    t0 = time.monotonic()
    outs = llm.generate(prompts, sampling())
    dt = time.monotonic() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"arm": "offline", "wall_s": round(dt, 1), "completion_tokens": toks,
            "tok_s": round(toks / dt, 1), "peak_rss_mb": peak_rss_mb(), "n": len(outs)}


async def _drain(engine, prompt: str, i: int, sp) -> int:
    final = None
    async for out in engine.generate(prompt, sp, request_id=f"r{i}"):
        final = out
    return len(final.outputs[0].token_ids)


async def arm_async(prompts: list[str], window: int | None) -> dict:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=MODEL, max_num_seqs=MAX_NUM_SEQS, gpu_memory_utilization=0.90))
    sp = sampling()
    sem = asyncio.Semaphore(window) if window else None
    t0 = time.monotonic()

    async def one(p: str, i: int) -> int:
        if sem is None:
            return await _drain(engine, p, i, sp)
        async with sem:
            return await _drain(engine, p, i, sp)

    toks = sum(await asyncio.gather(*[one(p, i) for i, p in enumerate(prompts)]))
    dt = time.monotonic() - t0
    name = "windowed" if window else "naive"
    result = {"arm": name, "wall_s": round(dt, 1), "completion_tokens": toks,
              "tok_s": round(toks / dt, 1), "peak_rss_mb": peak_rss_mb(),
              "n": len(prompts), "window": window}
    engine.shutdown()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()

    if args.arm:  # child: run one arm, emit one JSON line
        prompts = load_prompts()
        if args.arm == "offline":
            r = arm_offline(prompts)
        elif args.arm == "naive":
            r = asyncio.run(arm_async(prompts, window=None))
        else:
            r = asyncio.run(arm_async(prompts, window=2 * MAX_NUM_SEQS))
        print("SPIKE_RESULT " + json.dumps(r), flush=True)
        return

    # parent: one subprocess per arm (clean VRAM), collect + summarize
    results = []
    for arm in ARMS:
        print(f"=== arm: {arm} ===", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--arm", arm],
                              capture_output=True, text=True, timeout=1800)
        for line in proc.stdout.splitlines():
            if line.startswith("SPIKE_RESULT "):
                results.append(json.loads(line[len("SPIKE_RESULT "):]))
                print(line, flush=True)
                break
        else:
            print(f"ARM_FAILED {arm} rc={proc.returncode}\n{proc.stderr[-3000:]}", flush=True)
    print("SPIKE_SUMMARY " + json.dumps({
        "model": MODEL, "n_prompts": N_PROMPTS, "max_tokens": MAX_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS, "results": results}), flush=True)


if __name__ == "__main__":
    main()
