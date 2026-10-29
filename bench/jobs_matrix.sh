#!/usr/bin/env bash
# One defaults-matrix cell inside one HF Job: start the engine, run bench/matrix.py, upload the
# JSON lines. bench/matrix_cells.py launches it with the env below; to run one cell by hand:
#
#   hf jobs run --flavor a10g-small --timeout 90m --secrets HF_TOKEN \
#     -e CELL=small-short-vllm -e MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#     -e ENGINE_CMD="vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000" -e CELL_ARGS="--kind chat" \
#     vllm/vllm-openai:latest bash -c "curl -sLo /tmp/m.sh \
#       https://raw.githubusercontent.com/davanstrien/saturate/main/bench/jobs_matrix.sh && bash /tmp/m.sh"
#
# env: CELL, MODEL, ENGINE_CMD, CELL_ARGS; REF (saturate under test, default origin/main);
# RESULTS_URI (default hf://buckets/davanstrien/saturate-bench/matrix).
set -euo pipefail
REF="${REF:-origin/main}"
REPO_URL="${REPO_URL:-https://github.com/davanstrien/saturate}"
RESULTS_URI="${RESULTS_URI:-hf://buckets/davanstrien/saturate-bench/matrix}"

export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null && command -v curl >/dev/null \
  || { apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null; }
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="${HOME:-/root}/.local/bin:$PATH" UV_NO_CONFIG=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash -c "$ENGINE_CMD" > /tmp/engine.log 2>&1 < /dev/null &
git clone -q "$REPO_URL" /work/saturate && cd /work/saturate
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
git checkout -q "$REF"
SHA=$(git rev-parse --short HEAD)
echo "saturate at $SHA"
uv venv -q --python 3.12 /work/v && uv pip install -q --python /work/v/bin/python ".[hf]" pillow

n=0
until curl -sf localhost:8000/health >/dev/null; do
  sleep 5; n=$((n + 1))
  [ $n -gt 360 ] && { echo "engine not ready"; tail -c 6000 /tmp/engine.log; exit 1; }
done
echo "engine ready"

out="/tmp/${CELL}.jsonl"
# shellcheck disable=SC2086  # CELL_ARGS is a list of flags
/work/v/bin/python bench/matrix.py --cell "$CELL" --endpoint http://127.0.0.1:8000/v1 --model "$MODEL" \
  --label "$SHA" --out "$out" $CELL_ARGS || { tail -c 4000 /tmp/engine.log; exit 1; }

/work/v/bin/python bench/matrix_cells.py summarise "$out"
if [ -n "${HF_TOKEN:-}" ]; then
  /work/v/bin/python - "$out" "$RESULTS_URI" "${JOB_ID:-local}" <<'PY'
import sys
from huggingface_hub import HfFileSystem
path, uri, job = sys.argv[1:4]
dest = f"{uri}/{job}-{path.rsplit('/', 1)[-1]}"
HfFileSystem().put(path, dest)
print("results:", dest)
PY
fi
