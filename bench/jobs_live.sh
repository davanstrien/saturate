#!/usr/bin/env bash
# bench/live_checks.py against one real engine inside one HF Job: the signals check (probe,
# capacity, gauges) and the four row-count checks. ENGINE picks the engine and its image:
#
#   ENGINE=sglang; IMAGE=lmsysorg/sglang:latest
#   hf jobs run --flavor a10g-small --secrets HF_TOKEN -e ENGINE=$ENGINE -e REF=origin/main \
#     $IMAGE bash -c "curl -sLo /tmp/jobs_live.sh \
#       https://raw.githubusercontent.com/davanstrien/saturate/main/bench/jobs_live.sh && bash /tmp/jobs_live.sh"
#
# Images: vllm/vllm-openai:latest, lmsysorg/sglang:latest, ghcr.io/ggml-org/llama.cpp:server-cuda,
# ghcr.io/huggingface/text-embeddings-inference:86-latest (an A10G is compute capability 8.6).
# The engine starts with the flags EXPECT in live_checks.py assumes (llama.cpp: --parallel 8).
# Results: the JSON lines in the Job log, and in RESULTS_URI when HF_TOKEN is set.
set -uo pipefail
ENGINE="${ENGINE:?vllm, sglang, llamacpp or tei}"
REF="${REF:-origin/main}"
REPO_URL="${REPO_URL:-https://github.com/davanstrien/saturate}"
RESULTS_URI="${RESULTS_URI:-hf://buckets/davanstrien/saturate-bench}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
KIND=chat

export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null && command -v curl >/dev/null \
  || { apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null; }
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="${HOME:-/root}/.local/bin:$PATH" UV_NO_CONFIG=1
git clone -q "$REPO_URL" /work/saturate && cd /work/saturate
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
git checkout -q "$REF" && echo "saturate at $(git rev-parse --short HEAD)"
uv venv -q --python 3.12 /work/v && uv pip install -q --python /work/v/bin/python ".[hf]"

case "$ENGINE" in
  vllm)     ENGINE_START="vllm serve $MODEL --port 8000 --host 127.0.0.1";;
  sglang)   ENGINE_START="python3 -m sglang.launch_server --model-path $MODEL --port 8000 --host 127.0.0.1 --mem-fraction-static 0.7 --max-running-requests 256 --enable-metrics";;
  llamacpp) ENGINE_START="env LD_LIBRARY_PATH=/app /app/llama-server --hf-repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --hf-file qwen2.5-0.5b-instruct-q8_0.gguf --port 8000 --host 127.0.0.1 -ngl 99 --parallel 8 -c 32768 --metrics";;
  tei)      MODEL="BAAI/bge-small-en-v1.5"; KIND=embeddings
            ENGINE_START="text-embeddings-router --model-id $MODEL --port 8000 --hostname 127.0.0.1";;
  *)        echo "unknown ENGINE $ENGINE"; exit 2;;
esac
export ENGINE_START ENGINE_PGID_FILE=/tmp/engine.pgid
OUT=/work/out; mkdir -p "$OUT"

# its own process group (setsid), so the restart check can stop the engine and its workers
setsid bash -c "exec $ENGINE_START" > "$OUT/engine.log" 2>&1 < /dev/null &
echo $! > "$ENGINE_PGID_FILE"
n=0
until curl -sf localhost:8000/health >/dev/null; do
  sleep 5; n=$((n + 1))
  [ $n -gt 180 ] && { echo "engine not healthy"; tail -60 "$OUT/engine.log"; exit 1; }
done
echo "engine $ENGINE ready; /metrics HTTP $(curl -s -o /dev/null -w '%{http_code}' localhost:8000/metrics)"

/work/v/bin/python bench/live_checks.py --endpoint http://127.0.0.1:8000/v1 --model "$MODEL" \
  --engine "$ENGINE" --kind "$KIND" 2>"$OUT/checks.stderr" | tee "$OUT/checks.jsonl"
cp /tmp/engine-restart.log "$OUT/" 2>/dev/null || true

echo "##### summary"
/work/v/bin/python - "$OUT/checks.jsonl" <<'PY'
import json, sys
records = [json.loads(line) for line in open(sys.argv[1]) if line.startswith("{")]
for r in records:
    print(f"{r['check']:15} {'PASS' if r['pass'] else 'FAIL'}  {r.get('failures') or ''}")
print(f"{sum(r['pass'] for r in records)}/{len(records)} passed")
PY

if [ -n "${HF_TOKEN:-}" ]; then
  uv run -q --no-project --with huggingface_hub python - "$OUT" "$RESULTS_URI" <<'PY'
import os, sys
from huggingface_hub import HfFileSystem
out, uri = sys.argv[1], sys.argv[2]
dest = f"{uri}/live-{os.environ['ENGINE']}-{os.environ.get('JOB_ID', 'local')}"
fs = HfFileSystem()
for name in os.listdir(out):
    fs.put(os.path.join(out, name), f"{dest}/{name}")
print(f"results: {dest}/")
PY
fi
