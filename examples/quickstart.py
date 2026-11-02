# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "saturate[hf]==0.3.0",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.22.1"
# flavor = "a10g-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""OCR a dataset of page images with LightOnOCR-2-1B, into a new private dataset.

Seven pages of a NASA booklet, on HF Jobs (`hf` CLI 1.32 or later). The header above sets
the image that provides `vllm`, the GPU, the timeout and the secret; flags override it:

    hf jobs uv run \\
        https://raw.githubusercontent.com/davanstrien/saturate/main/examples/quickstart.py \\
        uv-scripts/ocr-demo your-username/ocr-demo-results

Your own dataset: change the first argument, and try `--max-samples 10` first. Kill the
Job at any point: the same command skips the pages already written.

Read the result with `load_dataset("your-username/ocr-demo-results", split="train")`. Each row
has the input row's `id`, the page's `markdown` and token counts, or an `error`. Successes
and errors are both kept, so filter on `error`.

For more OCR models, see https://huggingface.co/datasets/uv-scripts/ocr
"""

import argparse
import base64
import io

import pyarrow as pa
from PIL import Image, ImageOps

from saturate import Auto, Engine, dataset_rows, pump

MODEL = "lightonai/LightOnOCR-2-1B"
# The model card's serve flags: OCR never reuses an image, so these caches only cost memory.
# An 8192 context fits a 1540 px page and 4096 tokens of output, on half the GPU memory.
SERVE_ARGS = [
    "--max-model-len",
    "8192",
    "--limit-mm-per-prompt",
    '{"image": 1}',
    "--mm-processor-cache-gb",
    "0",
    "--no-enable-prefix-caching",
]
LONGEST_EDGE = 1540  # the resolution the model was trained on

# Declared, so that every part has the same columns and load_dataset reads the output.
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("error", pa.string()),
        ("markdown", pa.string()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
    ]
)


def to_data_url(image: Image.Image) -> str:
    """Fit the page to LONGEST_EDGE and return it as a base64 PNG."""
    page = ImageOps.contain(image.convert("RGB"), (LONGEST_EDGE, LONGEST_EDGE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dataset", help="Hub dataset with an image column")
    parser.add_argument("output_dataset", help="Hub dataset to write to (created private if missing)")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None, help="Process only the first N rows")
    args = parser.parse_args()

    rows = dataset_rows(
        args.input_dataset, split=args.split, columns=[args.image_column], limit=args.max_samples
    )

    def to_request(row: dict) -> dict:
        image_part = {"type": "image_url", "image_url": {"url": to_data_url(row[args.image_column])}}
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": [image_part]}],  # this model takes no text prompt
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
        }

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {
            "markdown": body["choices"][0]["message"]["content"].strip(),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    with Engine(MODEL, engine="vllm", extra_args=SERVE_ARGS) as endpoint:
        stats = pump(
            rows,
            to_request,
            parse,
            endpoint,
            f"hf://datasets/{args.output_dataset}/data",
            window=Auto(max_limit=48),  # a ceiling, not a setting: images in flight live in RAM
            schema=SCHEMA,
        )
    url = f"https://huggingface.co/datasets/{args.output_dataset}"
    print(f"{url}: {stats.rows_processed} ok, {stats.rows_failed} failed")


if __name__ == "__main__":
    main()
