# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "saturate[hf]",
# ]
# ///
"""Quickstart: data -> model -> nicer data, one GPU Job.

100 dolly instructions through Qwen2.5-0.5B, out as a private dataset repo
under your account. Kill it anytime; re-running resumes exactly.

On HF Jobs (vllm comes from the image; a few minutes on an A10G):

    hf jobs uv run --image vllm/vllm-openai:latest --flavor a10g-small \\
        --secrets HF_TOKEN examples/quickstart.py

Locally: `uv run examples/quickstart.py` on any machine with a GPU and
`vllm` on PATH (e.g. inside the same image).
"""

from huggingface_hub import HfApi

from saturate import Auto, Engine, dataset_rows, pump

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    api = HfApi()
    repo = f"{api.whoami()['name']}/saturate-quickstart"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    rows = dataset_rows(
        "databricks/databricks-dolly-15k",
        columns=["instruction"],
        ids="content",
        limit=100,
    )

    def to_request(row: dict) -> dict:
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": row["instruction"]}],
            "max_tokens": 150,
        }

    def parse(row: dict, body: dict) -> dict:
        return {
            "instruction": row["instruction"],
            "response": body["choices"][0]["message"]["content"],
        }

    with Engine(MODEL, engine="vllm") as endpoint:
        stats = pump(
            rows,
            to_request,
            parse,
            endpoint,
            f"hf://datasets/{repo}/data",
            window=Auto(initial=8),
        )
    print(f"done: {stats.rows_processed} rows -> hf://datasets/{repo}/data")


if __name__ == "__main__":
    main()
