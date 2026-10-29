"""One defaults-matrix cell against a running engine: Auto() a few times, then a sweep of Fixed
windows, on the same rows. Prints and appends one JSON line per run to --out.

    python bench/matrix.py --cell small-short-vllm --endpoint http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen2.5-0.5B-Instruct --out cell.jsonl

The score of a cell is Auto's rows/s over the best Fixed window's rows/s (bench/matrix_cells.py
summarises a results directory). Workloads (--kind):

- chat: one short prompt per row, --max-tokens out.
- long-prompt: a prompt of about --prompt-words words per row, --max-tokens out.
- image: one generated image per row (a data URL) and a one-line caption request.
- embeddings: one short passage per row to /embeddings.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import statistics
import tempfile
import time

from saturate import Auto, Fixed, pump

WORDS = ("archive ledger river harbour engine signal library county parish volume letter map "
         "survey census railway market council school chapel bridge mill canal station").split()


def make_rows(args, label: str, rep: int) -> list[dict]:
    rows = []
    for i in range(args.rows):
        row = {"id": f"{label}-{rep}-{i}", "i": i}
        if args.kind == "long-prompt":
            words = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(args.prompt_words)]
            row["text"] = f"Summarise in one sentence (item {i}): " + " ".join(words)
        elif args.kind == "embeddings":
            row["text"] = f"A short passage about the number {i}, for an embedding benchmark."
        else:
            row["text"] = f"Write one sentence about the number {i}."
        rows.append(row)
    return rows


def image_data_url(i: int, size: int) -> str:
    """A small generated PNG: different per row, so no image cache serves it twice."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
    draw = ImageDraw.Draw(image)
    draw.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], outline=(255, 255, 255), width=4)
    draw.text((size // 3, size // 2), str(i), fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def make_to_request(args):
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}

    def to_request(row: dict) -> dict:
        if args.kind == "embeddings":
            return {"model": args.model, "input": row["text"]}
        if args.kind == "image":
            content = [{"type": "image_url", "image_url": {"url": image_data_url(row["i"], args.image_size)}},
                       {"type": "text", "text": "Describe this image in one line."}]
        else:
            content = row["text"]
        return {"model": args.model, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": content}], **extra}

    return to_request


def make_parse(args):
    def parse(row: dict, body: dict) -> dict:
        if args.kind == "embeddings":
            return {"dim": len(body["data"][0]["embedding"])}
        usage = body.get("usage") or {}
        return {"out_tokens": usage.get("completion_tokens")}

    return parse


def telemetry_summary(out_dir: str) -> dict:
    ticks = []
    for path in glob.glob(f"{out_dir}/**/telemetry-*.jsonl", recursive=True):
        with open(path) as f:
            ticks += [json.loads(line) for line in f if line.strip()]
    kv = [t["kv"] for t in ticks if t.get("kv") is not None]
    preempts = [t["preempts"] for t in ticks if t.get("preempts") is not None]
    limits = [t["limit"] for t in ticks if t.get("limit") is not None]
    return {"ticks": len(ticks), "kv_max": max(kv, default=None),
            "kv_mean": round(statistics.mean(kv), 3) if kv else None,
            "preempts": preempts[-1] - preempts[0] if preempts else None,
            "limit_max": max(limits, default=None)}


def one_run(args, label: str, window, rep: int) -> dict:
    out_dir = tempfile.mkdtemp()
    route = "/embeddings" if args.kind == "embeddings" else "/chat/completions"
    t0 = time.time()
    stats = pump(make_rows(args, label, rep), make_to_request(args), make_parse(args),
                 endpoint=args.endpoint, output=out_dir, window=window, route=route,
                 prepare_workers=args.prepare_workers)
    wall = time.time() - t0
    record = {"cell": args.cell, "model": args.model, "kind": args.kind, "arm": label, "rep": rep,
              "rows": args.rows, "ok": stats.rows_processed, "failed": stats.rows_failed,
              "wall_s": round(wall, 1), "rows_per_s": round(stats.rows_processed / wall, 2),
              "tok_s": stats.tokens_per_sec, "final_limit": stats.final_limit,
              "cut_reasons": stats.cut_reasons, "bound_by": stats.bound_by, "hints": stats.hints,
              "saturate": args.label, **telemetry_summary(out_dir)}
    print(json.dumps(record), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["chat", "long-prompt", "image", "embeddings"], default="chat")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-words", type=int, default=1500)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--prepare-workers", type=int, default=0, help="threads for to_request (image: 4)")
    ap.add_argument("--no-think", action="store_true", help="chat_template_kwargs enable_thinking=False")
    ap.add_argument("--auto-reps", type=int, default=3)
    ap.add_argument("--fixed", default="16,32,64,128,256,512", help="Fixed windows to sweep, comma-separated")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--label", default="", help="recorded as `saturate`: the version under test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.no_warmup:
        one_run(args, "warmup", Fixed(32), 0)
    for rep in range(args.auto_reps):
        one_run(args, "auto", Auto(), rep)
    for window in [int(x) for x in args.fixed.split(",") if x]:
        one_run(args, f"fixed-{window}", Fixed(window), 0)


if __name__ == "__main__":
    main()
