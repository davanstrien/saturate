#!/usr/bin/env bash
# A/B benchmark against a real vLLM engine on HF Jobs: both saturate versions run inside the
# same Job, against the same engine, interleaved (bench/ab.py), so node-to-node variance
# cancels. Launch from a laptop:
#
#   hf jobs run --flavor a10g-large --secrets HF_TOKEN \
#     -e BASE=origin/main -e HEAD=origin/main -e REPS=3 \
#     vllm/vllm-openai:latest \
#     bash -c "curl -sL https://raw.githubusercontent.com/davanstrien/saturate/bench/harness/bench/jobs_ab.sh | bash"
#
# (`hf jobs run IMAGE cmd` replaces the image's entrypoint, so the vLLM server is started here.)
# Results: the markdown table in the Job log, and the JSON in RESULTS_URI when HF_TOKEN is set.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE="${BASE:-origin/main}"
HEAD="${HEAD:-origin/main}"
HARNESS_REF="${HARNESS_REF:-origin/bench/harness}"  # where bench/ itself comes from
REPS="${REPS:-3}"
SUITE="${SUITE:-vllm}"
REPO_URL="${REPO_URL:-https://github.com/davanstrien/saturate}"
RESULTS_URI="${RESULTS_URI:-hf://buckets/davanstrien/saturate-bench}"

command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }  # not in the vllm image
python3 -m pip install -q --root-user-action=ignore uv huggingface_hub
git clone -q "$REPO_URL" /work/saturate
cd /work/saturate
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
git checkout -q "$HARNESS_REF"

vllm serve "$MODEL" --port 8000 > /tmp/vllm.log 2>&1 &
python3 - <<'EOF'
import time, urllib.request
deadline = time.time() + 900
while time.time() < deadline:
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status == 200:
            print("vllm ready", flush=True)
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit("vllm did not become ready; see /tmp/vllm.log")
EOF

out="/tmp/ab-$(date +%Y%m%d-%H%M%S).json"
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
UV_NO_CONFIG=1 python3 bench/ab.py --base "$BASE" --head "$HEAD" --suite "$SUITE" --reps "$REPS" \
  --endpoint http://127.0.0.1:8000/v1 --model "$MODEL" --python "$PY_VERSION" --out "$out"

if [ -n "${HF_TOKEN:-}" ]; then
  python3 - "$out" "$RESULTS_URI" "${JOB_ID:-local}" <<'EOF'
import sys
from huggingface_hub import HfApi, HfFileSystem
path, uri, job = sys.argv[1:4]
bucket = uri.removeprefix("hf://buckets/").split("/")[:2]
HfApi().create_bucket("/".join(bucket), private=True, exist_ok=True)
HfFileSystem().put(path, f"{uri}/{job}-{path.rsplit('/', 1)[-1]}")
print(f"results: {uri}/{job}-{path.rsplit('/', 1)[-1]}")
EOF
fi
