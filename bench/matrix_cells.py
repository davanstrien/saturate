"""The defaults matrix: which cells exist, launching them on HF Jobs, and summarising results.

    python bench/matrix_cells.py list
    python bench/matrix_cells.py launch --tier small [--ref origin/main] [--only NAME,...] [--dry-run]
    python bench/matrix_cells.py summarise results/*.jsonl

A cell is one engine, one model, one workload on one GPU flavor. bench/jobs_matrix.sh runs it:
Auto() three times, then a sweep of Fixed windows. A cell's score is Auto's median rows/s over
the best Fixed window's rows/s. The README's starting points come from the cells where Auto
scores low: there, the best Fixed window is the documented start.

Tiers: `small` runs on one small GPU (about $1-2 a cell); `large` needs A100/H200 nodes.
"""

from __future__ import annotations

import argparse
import collections
import json
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, field

SCRIPT_URL = "https://raw.githubusercontent.com/davanstrien/saturate/{branch}/bench/jobs_matrix.sh"
VLLM_IMAGE = "vllm/vllm-openai:latest"
SGLANG_IMAGE = "lmsysorg/sglang:latest"
LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-latest"
SMALL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class Cell:
    name: str
    tier: str
    flavor: str
    image: str
    model: str
    engine_cmd: str
    args: list[str] = field(default_factory=list)


def vllm(model: str, *flags: str) -> str:
    return " ".join(["vllm serve", model, "--port 8000 --host 127.0.0.1", *flags])


CELLS: tuple[Cell, ...] = (
    # small text model, three engines, three output shapes
    Cell("small-short-vllm", "small", "a10g-small", VLLM_IMAGE, SMALL, vllm(SMALL), ["--kind", "chat"]),
    Cell("small-long-vllm", "small", "a10g-small", VLLM_IMAGE, SMALL, vllm(SMALL),
         ["--kind", "chat", "--max-tokens", "512", "--rows", "1500"]),
    Cell("small-prompt-vllm", "small", "a10g-small", VLLM_IMAGE, SMALL, vllm(SMALL),
         ["--kind", "long-prompt", "--rows", "2000"]),
    Cell("small-short-sglang", "small", "a10g-small", SGLANG_IMAGE, SMALL,
         f"python3 -m sglang.launch_server --model-path {SMALL} --port 8000 --host 127.0.0.1 "
         "--mem-fraction-static 0.7 --enable-metrics", ["--kind", "chat"]),
    Cell("small-short-llamacpp", "small", "a10g-small", LLAMACPP_IMAGE, SMALL,
         "env LD_LIBRARY_PATH=/app /app/llama-server --hf-repo Qwen/Qwen2.5-0.5B-Instruct-GGUF "
         "--hf-file qwen2.5-0.5b-instruct-q8_0.gguf --port 8000 --host 127.0.0.1 -ngl 99 "
         "--parallel 8 -c 32768 --metrics",
         ["--kind", "chat", "--rows", "1500", "--fixed", "4,8,16,32,64"]),
    # the same engines in steady state: about 2 minutes per run at the best window, so the
    # ramp from Auto's start is a small share (the cells above measure short runs)
    Cell("steady-short-vllm", "small", "a10g-small", VLLM_IMAGE, SMALL, vllm(SMALL),
         ["--kind", "chat", "--rows", "35000", "--auto-reps", "2", "--fixed", "128,256,512,1024"]),
    Cell("steady-long-vllm", "small", "a10g-small", VLLM_IMAGE, SMALL, vllm(SMALL),
         ["--kind", "chat", "--max-tokens", "512", "--rows", "30000", "--auto-reps", "2",
          "--fixed", "128,256,512,1024"]),
    Cell("steady-short-sglang", "small", "a10g-small", SGLANG_IMAGE, SMALL,
         f"python3 -m sglang.launch_server --model-path {SMALL} --port 8000 --host 127.0.0.1 "
         "--mem-fraction-static 0.7 --enable-metrics",
         ["--kind", "chat", "--rows", "30000", "--auto-reps", "2", "--fixed", "128,256,512,1024"]),
    # embeddings
    Cell("embed-tei", "small", "a10g-small", TEI_IMAGE, "BAAI/bge-small-en-v1.5",
         "text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8000 --hostname 127.0.0.1",
         ["--kind", "embeddings", "--rows", "20000", "--fixed", "32,64,128,256,512,1024"]),
    # images: one generated image per row, a multimodal model
    Cell("image-4b-vllm", "small", "a10g-large", VLLM_IMAGE, "Qwen/Qwen3.5-4B",
         vllm("Qwen/Qwen3.5-4B", "--max-model-len 8192"),
         ["--kind", "image", "--rows", "1500", "--no-think", "--prepare-workers", "4"]),
    # a mid-size model on one bigger GPU
    Cell("mid-short-vllm", "small", "l40sx1", VLLM_IMAGE, "Qwen/Qwen3.5-9B",
         vllm("Qwen/Qwen3.5-9B", "--max-model-len 8192"), ["--kind", "chat", "--no-think"]),
    # large models (the 2026-10-29 pilot cells, with more rows so the ramp is a small share)
    Cell("large-short-vllm", "large", "a100-large", VLLM_IMAGE, "Qwen/Qwen3.8-27B",
         vllm("Qwen/Qwen3.8-27B", "--max-model-len 8192"), ["--kind", "chat", "--no-think"]),
    Cell("large-long-vllm", "large", "a100-large", VLLM_IMAGE, "Qwen/Qwen3.8-27B",
         vllm("Qwen/Qwen3.8-27B", "--max-model-len 8192"),
         ["--kind", "chat", "--no-think", "--max-tokens", "512", "--rows", "1500"]),
    Cell("large-tp2-vllm", "large", "h200x2", VLLM_IMAGE, "Qwen/Qwen3.8-27B",
         vllm("Qwen/Qwen3.8-27B", "--max-model-len 8192 --tensor-parallel-size 2"),
         ["--kind", "chat", "--no-think", "--rows", "30000", "--fixed", "128,256,512,1024,2048"]),
)


def launch(cells: list[Cell], ref: str, branch: str, dry_run: bool) -> None:
    url = SCRIPT_URL.format(branch=branch)
    for cell in cells:
        cmd = ["hf", "jobs", "run", "-d", "--flavor", cell.flavor, "--timeout", "90m",
               "--secrets", "HF_TOKEN", "-e", f"CELL={cell.name}", "-e", f"MODEL={cell.model}",
               "-e", f"ENGINE_CMD={cell.engine_cmd}", "-e", f"CELL_ARGS={' '.join(cell.args)}",
               "-e", f"REF={ref}", "-e", f"HARNESS_REF=origin/{branch}", cell.image,
               "bash", "-c", f"curl -sLo /tmp/m.sh {url} && bash /tmp/m.sh"]
        if dry_run:
            print(shlex.join(cmd))
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        job = (result.stdout + result.stderr).strip().splitlines()[-1:] or ["?"]
        print(f"{cell.name:22} {cell.flavor:11} {job[0]}", flush=True)


def summarise(paths: list[str]) -> None:
    """Per cell: Auto's median rows/s, the best Fixed window, the ratio, where Auto settled."""
    runs = collections.defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    runs[record["cell"]].append(record)
    print(f"{'cell':22} {'auto rows/s':>11} {'best fixed':>16} {'ratio':>6}  auto settles  cuts")
    for cell, records in sorted(runs.items()):
        auto = [r for r in records if r["arm"] == "auto"]
        fixed = [r for r in records if r["arm"].startswith("fixed-")]
        if not auto or not fixed:
            print(f"{cell:22} incomplete ({len(auto)} auto, {len(fixed)} fixed runs)")
            continue
        auto_rate = statistics.median(r["rows_per_s"] for r in auto)
        best = max(fixed, key=lambda r: r["rows_per_s"])
        settles = sorted(r["final_limit"] for r in auto)
        cuts = collections.Counter()
        for r in auto:
            cuts.update(r.get("cut_reasons") or {})
        print(f"{cell:22} {auto_rate:11.1f} {best['arm']:>9} {best['rows_per_s']:6.1f} "
              f"{auto_rate / best['rows_per_s']:6.0%}  {settles}  {dict(cuts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    go = sub.add_parser("launch")
    go.add_argument("--tier", choices=["small", "large", "all"], default="small")
    go.add_argument("--only", default="", help="cell names, comma-separated")
    go.add_argument("--ref", default="origin/main", help="the saturate commit under test")
    go.add_argument("--branch", default="main", help="where jobs_matrix.sh is fetched from")
    go.add_argument("--dry-run", action="store_true")
    report = sub.add_parser("summarise")
    report.add_argument("paths", nargs="+")
    args = ap.parse_args()

    if args.command == "list":
        for cell in CELLS:
            print(f"{cell.name:22} {cell.tier:6} {cell.flavor:11} {cell.model:30} {' '.join(cell.args)}")
    elif args.command == "launch":
        names = {name for name in args.only.split(",") if name}
        cells = [c for c in CELLS if args.tier in ("all", c.tier) and (not names or c.name in names)]
        if not cells:
            sys.exit("no cell matches")
        launch(cells, args.ref, args.branch, args.dry_run)
    else:
        summarise(args.paths)


if __name__ == "__main__":
    main()
