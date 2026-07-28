# Codex review round 3 — triage (2026-07-28)

Third adversarial review: 28 findings. Independent verification against HEAD confirmed 26
valid, 1 partially valid (#22), 1 valid-by-design (#7). Constraint honored throughout: where
narrowing a CONTRACT claim was the honest fix, the claim was narrowed instead of adding code
(LOC ceiling held at 1100). Every finding has a disposition below — nothing dropped silently.

Suite and oracle (`ORACLE_ADAPTER=clean`, 9/9) green after every commit on the fix branch.

| # | Finding | Disposition |
|---|---|---|
| 1 | Dead breaker → durable error rows, skipped on resume | **FIXED**: `FatalTransportError` aborts the run; worker routes it past the row-error handler; drain flushes paid-for rows in a `finally`. Tested |
| 2 | Parquet flush takes schema from row 0, drops success columns | **FIXED**: rows normalized to the union of keys before table build. Reproduced, tested |
| 3 | FileSink path traversal via ids | **FIXED**: unsafe ids (separators, empty, dot-leading) raise ValueError. Tested |
| 4 | Non-dict parse result aborts run after API spend | **FIXED**: dict-validated at the parse call → healable error row. Tested |
| 5 | Cross-part schema instability (all-null / type-changed columns) | **FIXED within-run**: schema pinned at first flush, later parts cast (permissive unify); incompatible mid-run type change raises at flush. Cross-run stability stays a documented non-guarantee (CONTRACT §8). Tested |
| 6 | Engine process leaks when readiness fails | **FIXED**: failed boot reuses the `__exit__` kill ladder and re-raises. Tested |
| 7 | Readiness accepts any status <500 (404/400 count as ready) | **REJECTED (by design)**: a 400 proves the API path parses requests — health endpoints go 200 while the completion path is still dark. Workload-specific probes are wrapper-layer material |
| 8 | Completed-results buffer unbounded with a slow consumer | **DEFERRED (Aug 3)**: real; bounding the queue or holding admission credits until consumption interacts with the feed-ahead design — design-review material, not a patch |
| 9 | Blocking source/sink calls on the event loop | **DEFERRED (Aug 3)**: async source/sink protocol is post-v1 architecture (couples to the AsyncLLM transport decision 15) |
| 10 | Background tasks cancelled but never awaited (silent controller death) | **FIXED**: limiter awaits its tick task at exit (crash surfaces); `through` reaps feeder+workers via `gather`. Tested |
| 11 | Retry-After honored beyond the retry budget (3600s sleep vs 300s budget) | **FIXED**: sleeps capped at remaining budget. Tested |
| 12 | Multipart retries re-send consumed file objects (empty bodies) | **FIXED (narrowed)**: multipart is single-attempt. Seek-rewind for replayable bodies = post-v1 upgrade |
| 13 | tok/s divides by fixed 2s; `_best_tok` all-time max blocks recovery | **FIXED**: actual tick elapsed used; a cut decays `_best_tok` by half. Tested |
| 14 | Manifest without surviving part counts as done → silent data loss | **CONTRACT-NARROWED** (§3): resume trusts manifests; out-of-band part loss = rows neither reprocessed nor returned; both recovery paths documented. A code fix would break oracle test I (manifest-based resume probe). Optional loud missing-part warning in `read_output` = post-v1 (+~4 LOC) |
| 15 | FileSink writes not crash-atomic | **FIXED**: same-dir temp + `os.replace`. Tested |
| 16 | hf:// advertised but huggingface_hub not installable from this package | **FIXED**: `pumpjack[hf]` extra. The `makedirs` swallow stays (HfFileSystem dirs are implicit; auth errors surface at first flush) |
| 17 | Engine stdout corrupts the single Stats JSON stdout line | **FIXED**: engine stdout → stderr (fd 2). CONTRACT §7 holds |
| 18 | Completion marker written before stats/telemetry | **FIXED**: marker moved to the end of `pump()`, after sidecars; §5 documents marker-implies-sidecars. `drain` alone no longer writes markers (docstring'd) |
| 19 | Positional shard fan-out + per-process dedup misses cross-shard duplicates | **DEFERRED (Aug 3)**: reader rule already resolves duplicate ids; hash-partitioning by id vs documenting at-least-once overlap is a design choice |
| 20 | 64-bit content hash too short for global-uniqueness claims | **CONTRACT-CAVEATED** (§2): collision math documented; negligible at the ~10M-row v1 scale, widen or supply ids beyond ~5e9 rows |
| 21 | `read_output` materializes everything, silently skips unreadable parts | **DEFERRED (Aug 3)**: materialization already documented; strict-mode + streaming reads couple to the publish() compile step |
| 22 | Retry classification asymmetries (408/425 poison, 429 never a breaker event…) | **DEFERRED (Aug 3)**: verified as asymmetries, not correctness bugs; revisit as an explicit retry-status table with the Pacer (decision 14) |
| 23 | Weak public validation (Fixed(0) deadlock, Path output as custom sink) | **FIXED**: `Fixed(<1)` raises; `as_sink` accepts Path. Tested. (Invalid shard geometry / request discriminants remain lax — post-v1 polish) |
| 24 | Telemetry filename 1s resolution overwrites same-shard runs | **FIXED**: uuid suffix; CONTRACT §1/§6 pattern updated |
| 25 | `pump()` cannot run inside a running event loop | **DEFERRED (Aug 3)**: `_pump` exists and is importable; deciding the public `apump()` surface is API-design, not a patch |
| 26 | flush_every=10 → 2 remote objects per 10 rows | **DEFERRED (Aug 3)**: byte/time/row flush policy is backend-aware design work; interim guidance = raise `flush_every` on remote sinks |
| 27 | LOC ceiling pressures reliability work; make it advisory | **REJECTED**: the ceiling stayed hard and every fix above landed inside it (1100/1100). It is doing its job |
| 28 | Internal quotes/names/paths in docs; leak-guard violation | **FIXED**: decisions.md + why.md scrubbed to neutral attributions (this commit); originals live in internal notes and git history — pre-public checklist now requires history squash/rewrite |

## Round 4 addendum (PR #1 review, 2026-07-28)

The fix-diff re-review confirmed most of round 3 and raised 6 blockers + 6 gaps. Dispositions:

| Blocker | Disposition |
|---|---|
| 1 Cross-part schema breaks in the null-first order | **FIXED**: null-typed columns default to string at flush (replaces the error-only pin — same rule, all columns). Reverse-order test added |
| 2 Retry budget not a deadline (post-sleep request) | **FIXED**: budget rechecked after every backoff; test asserts request count. Per-attempt timeout capping = deferred (needs timeout plumbing through client.post) |
| 3 `_best_tok` decay regrows at min limit | **FIXED**: decay only when the cut actually reduces the limit. Tested |
| 4 Marker ⇒ sidecars overstated | **CONTRACT-NARROWED** (§5): marker guarantees ordering (writes attempted), not sidecar presence |
| 5 Teardown reaps only the group leader | **PARTIAL**: leader reaped after SIGKILL. Full group-membership verification (retain PGID, probe, escalate, handle races) = deferred; on Jobs the container teardown owns stragglers |
| 6 Crashed tick loop leaks the HTTP client | **FIXED**: `aclose()` in a finally. Tested |

Gaps: Auto bound validation **FIXED** (tested); CONTRACT "every admitted row" invariant
**NARROWED** (fatal-abort in-flight rows produce no record, re-admitted next run — in
resume's favor); README marker/`[hf]` claims **FIXED**. Deferred: workload-aware readiness
probe (still by-design, wrapper-layer material), Arrow-serializability validation of parse
values, FileSink ext validation + concurrent-duplicate tmp races.

LOC ceiling renegotiated 1100 → 1120 for this round (4th renegotiation, decisions.md).

## Design direction (review's closing suggestion)

Typed fatal-vs-row-error outcome: **adopted** (#1). Frozen output schema: adopted within-run
(#5); a versioned run descriptor with part integrity metadata remains Aug 3 material together
with #8/#9 (bounded structured concurrency, async sink/source). Controller
recent-baseline-vs-all-time-max: partially adopted (#13's decay); full epoch-baseline
redesign deferred to the calibration-grid work.
