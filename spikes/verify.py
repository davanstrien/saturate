"""Verify a pumpjack output dir on hf://datasets from the laptop — CONTRACT.md rules.

    python3 verify.py davanstrien/pumpjack-embed-4job [expected_ids]
"""

import sys
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

repo = sys.argv[1]
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None

api = HfApi()
files = api.list_repo_files(repo, repo_type="dataset")
parts = [f for f in files if f.startswith("data/part-") and f.endswith(".parquet")]
manifests = [f for f in files if f.startswith("data/_manifest/ids-") and f.endswith(".parquet")]
markers = sorted(f for f in files if "/completions/" in f and f.endswith(".done"))
telemetry = [f for f in files if "telemetry-shard" in f]

print(f"repo={repo}")
print(f"parts={len(parts)} manifests={len(manifests)} markers={len(markers)} telemetry={len(telemetry)}")
print(f"marker names: {[m.split('/')[-1] for m in markers]}")

ids, errors, n_texts_total, tok_total = [], 0, 0, 0
for m in manifests:
    p = hf_hub_download(repo, m, repo_type="dataset")
    t = pq.read_table(p, columns=["id", "error"])
    ids.extend(t.column("id").to_pylist())
    errors += sum(1 for e in t.column("error").to_pylist() if e is not None)

c = Counter(ids)
dupes = {k: v for k, v in c.items() if v > 1}
print(f"records={len(ids)} unique={len(c)} dupes={len(dupes)} error_rows={errors}")
if dupes:
    print(f"  DUPE SAMPLE: {list(dupes.items())[:5]}")
if expected:
    print(f"EXPECTED {expected}: unique {'OK' if len(c) == expected else 'MISMATCH'}, "
          f"dupes {'OK' if not dupes else 'FAIL'}")

# spot-check one data part for embedding shape
if parts:
    p = hf_hub_download(repo, parts[0], repo_type="dataset")
    t = pq.read_table(p)
    print(f"part[0] cols={t.column_names} rows={t.num_rows}")
    if "embeddings" in t.column_names:
        row0 = t.column("embeddings")[0].as_py()
        print(f"part[0] row0: n_embeddings={len(row0)} dim={len(row0[0]) if row0 else 0}")
    if "n_texts" in t.column_names:
        print(f"part[0] n_texts sum={sum(x for x in t.column('n_texts').to_pylist() if x)}")
