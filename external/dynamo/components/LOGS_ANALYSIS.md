# Dynamo Router Logs Analysis Guide

This document is a self-contained reference for extracting and analyzing logs from a
running or completed Dynamo experiment.  It covers four data sources:

1. **Docker container logs** — real-time routing decisions, NATS overhead, KV opportunity cost
2. **`processor_overhead.jsonl`** — structured JSON Lines file (one record per request)
3. **NAT profiler CSVs** — per-LLM-call TTFT / ITL / TPS from the benchmark runner
4. **Prometheus / Grafana** — live and historical system metrics

---

## 0. Context: Two Startup Modes

| Script | Router | Custom Processor? | Overhead Log |
|--------|--------|-------------------|--------------|
| `start_dynamo_unified_vllm.sh` | Built-in KV (default) | No | None |
| `start_dynamo_optimized_thompson_hints_vllm.sh` | Thompson Sampling | Yes | `/workspace/logs/processor_overhead.jsonl` |

Both scripts mount a host directory to `/workspace/logs` inside the container.
The host path defaults to `external/dynamo/logs/` (next to the script) and can be
overridden with `DYNAMO_LOGS_DIR=/path/to/logs`.

---

## 1. Docker Container Logs

### 1.1 Live tail
```bash
docker logs -f dynamo-vllm 2>&1
```

### 1.2 Save to file for offline analysis
```bash
docker logs dynamo-vllm 2>&1 > /tmp/dynamo_run.log
```

### 1.3 Key log prefixes (Thompson router only)

| Prefix | What it shows |
|--------|--------------|
| `processor.generate: Processing request:` | Request arrived at processor — prefix_id, osl, iat, token count |
| `processor._pick_worker: nats_overhead:` | NATS RTT to router (ms) |
| `processor.generate: Routing decision:` | Worker chosen + `routing_ms` + `proc_overhead_ms` |
| `router._select_worker: kv_thompson:` | Full per-worker scoring — `ov`, `q`, `ld`, `sc`, `p` |
| `router._select_worker: kv_thompson chosen_breakdown:` | Score components for the chosen worker |
| `router._select_worker: kv_opportunity:` | KV delta vs best-idle worker, extra cold tokens |
| `router.generate: Router picked` | Chosen worker + `consecutive` call count |
| `router.feedback: Feedback:` | `elapsed_ms` + `consecutive` + reward |
| `router._sweep_pending: Timeout feedback:` | Requests that never returned (worker crash / timeout) |

---

## 2. NATS Overhead Analysis (Thompson only)

The `nats_overhead` log line is emitted once per request inside `_pick_worker`.
It measures the round-trip time from sending the routing RPC to receiving the worker ID back.

### 2.1 Quick terminal summary
```bash
docker logs dynamo-vllm 2>&1 | grep "nats_overhead:" | \
  python3 -c "
import sys, re
vals = [float(m.group(1)) for l in sys.stdin if (m := re.search(r'nats_rtt_ms=([\d.]+)', l))]
s = sorted(vals); n = len(s)
print(f'n={n}  median={s[n//2]:.1f}ms  P90={s[int(0.9*n)]:.1f}ms  P95={s[int(0.95*n)]:.1f}ms  max={max(s):.1f}ms')
"
```

### 2.2 Pandas analysis from the JSON Lines file
```python
import pandas as pd

# Load — one row per routed request
df = pd.read_json("external/dynamo/logs/processor_overhead.jsonl", lines=True)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

# Fields available:
#   ts, prefix_id, worker_id, decision_id, tokens_in,
#   osl, iat, reuse_budget, is_auto,
#   nats_rtt_ms     — pure NATS round-trip (ms)
#   proc_overhead_ms — total pre-engine overhead including hint extraction + prefix state

print(df["nats_rtt_ms"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

# Overhead over time (rolling 30-second median)
df.set_index("ts")["nats_rtt_ms"].resample("30s").median().plot(title="NATS RTT over time (ms)")

# Does NATS RTT correlate with token count?
df.plot.scatter(x="tokens_in", y="nats_rtt_ms", alpha=0.3, title="NATS RTT vs prompt length")

# Routing fallback rate (no worker returned)
print("Routing failures:", (~df["routing_routed"]).sum(), "/", len(df))
```

---

## 3. Routing Decision Analysis (Thompson only)

### 3.1 Queue depth distribution at decision time
```bash
docker logs dynamo-vllm 2>&1 | grep "kv_thompson:" | \
  python3 -c "
import sys, re
lines = sys.stdin.readlines()
worker_queues = {}
load_mods_all = []
for line in lines:
    for wid, ov, q, ld, sc in re.findall(r'w(\d+)=\[ov=([\d.]+) q=(\d+) ld=([\d.]+) sc=([\d.]+)', line):
        worker_queues.setdefault(wid[-5:], []).append(int(q))
        load_mods_all.append(float(ld))
near_zero = sum(1 for x in load_mods_all if x < 0.002)
print(f'load_mod near-zero (<0.002): {near_zero}/{len(load_mods_all)} ({100*near_zero/max(len(load_mods_all),1):.0f}%)')
print()
print('Per-worker avg queue depth:')
for w, qs in sorted(worker_queues.items(), key=lambda x: -sum(x[1])/len(x[1])):
    print(f'  w...{w}: avg={sum(qs)/len(qs):.1f}  p90={sorted(qs)[int(0.9*len(qs))]:.0f}  max={max(qs)}')
"
```

### 3.2 Score component breakdown (chosen worker)
```bash
docker logs dynamo-vllm 2>&1 | grep "chosen_breakdown:" | \
  python3 -c "
import sys, re
lines = sys.stdin.readlines()
bases, lints_v, affs, qs = [], [], [], []
for l in lines:
    m = re.search(r'q=(\d+) base=([+-][\d.]+) lints=([+-][\d.]+) affinity=([+-][\d.]+)', l)
    if m:
        qs.append(int(m.group(1))); bases.append(float(m.group(2)))
        lints_v.append(float(m.group(3))); affs.append(float(m.group(4)))
n = len(bases) or 1
print(f'Decisions: {n}')
print(f'  avg score_base  : {sum(bases)/n:.4f}  (base=0 means KV signal erased by queue penalty)')
print(f'  avg lints       : {sum(lints_v)/n:.4f}')
print(f'  avg affinity    : {sum(affs)/n:.4f}')
print(f'  base=0 rate     : {100*sum(1 for b in bases if abs(b)<0.002)/n:.0f}%')
print(f'  queue@chosen p50: {sorted(qs)[n//2]}  p90: {sorted(qs)[int(0.9*n)]}  max: {max(qs)}')
"
```

### 3.3 KV opportunity cost (QUEUED vs idle capacity)
```bash
docker logs dynamo-vllm 2>&1 | grep "kv_opportunity: QUEUED_PREFERRED" | \
  python3 -c "
import sys, re
lines = sys.stdin.readlines()
print(f'QUEUED_PREFERRED decisions: {len(lines)}')
deltas = [float(m.group(1)) for l in lines if (m := re.search(r'kv_delta=([+-][\d.]+)', l))]
tokens = [float(m.group(1)) for l in lines if (m := re.search(r'extra_cold_tokens=([\d.]+)', l))]
if deltas:
    print(f'  avg KV delta (chosen - idle): {sum(deltas)/len(deltas):+.3f}')
    print(f'  avg extra cold tokens saved:  {sum(tokens)/len(tokens):.0f}')
"
```

### 3.4 Consecutive same-worker routing & elapsed time from feedback
```bash
docker logs dynamo-vllm 2>&1 | grep "Feedback:" | \
  python3 -c "
import sys, re
lines = sys.stdin.readlines()
elapsed = [float(m.group(1)) for l in lines if (m := re.search(r'elapsed_ms=([\d.]+)', l))]
consec  = [int(m.group(1))   for l in lines if (m := re.search(r'consecutive=(\d+)', l))]
if elapsed:
    s = sorted(elapsed); n = len(s)
    print(f'Feedback elapsed_ms (n={n}):')
    print(f'  median={s[n//2]:.0f}  P90={s[int(0.9*n)]:.0f}  P95={s[int(0.95*n)]:.0f}  max={max(s):.0f}')
    print(f'  >10s: {sum(1 for x in elapsed if x>10000)} ({100*sum(1 for x in elapsed if x>10000)/n:.0f}%)')
    c = sorted(consec); nc = len(c)
    print(f'Consecutive same-worker (n={nc}): median={c[nc//2]}  max={max(c)}')
"
```

---

## 4. Profiler TTFT / ITL / TPS Analysis (Benchmark Output CSVs)

The benchmark runner generates comparison CSVs at:
```
examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/<experiment>/comparison/
  throughput_histogram_per_llm_call_data.csv   ← per-LLM-call TTFT, TPS, ITL
  throughput_histogram_data.csv                 ← per-request median metrics
  throughput_histogram_raw_itl_data.csv         ← raw token-to-token ITL gaps
```

### 4.1 Quick stats (no Python needed)
```bash
BASE=examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals
EXP=multi_domain_kv_thompson_9   # ← change to your experiment name

~/.venvs/nat_dynamo_eval/bin/python3 << 'EOF'
import csv, numpy as np, os
import sys

exp = os.environ.get("EXP", "multi_domain_kv_thompson_9")
base = f"examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/{exp}"

for fname, label in [
    ("comparison/throughput_histogram_per_llm_call_data.csv", "Per-LLM-call"),
    ("comparison/throughput_histogram_data.csv", "Per-request"),
]:
    fp = f"{base}/{fname}"
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        rows = list(csv.DictReader(f))

    if label == "Per-LLM-call":
        ttft = sorted([float(r["ttft_ms"]) for r in rows if r.get("ttft_ms")])
        tps  = sorted([float(r["tps"])     for r in rows if r.get("tps")])
        itl  = sorted([float(r["itl_ms"])  for r in rows if r.get("itl_ms") and r["itl_ms"] != "nan"])
        n = len(ttft)
        p = lambda v, pct: v[int(pct*len(v)/100)]
        print(f"\n=== {label} (n={n}) ===")
        print(f"  TTFT  mean={np.mean(ttft):.0f}  P50={p(ttft,50):.0f}  P90={p(ttft,90):.0f}  P95={p(ttft,95):.0f}  ms")
        print(f"  TPS   mean={np.mean(tps):.1f}   P50={p(tps,50):.1f}   P90={p(tps,90):.1f}")
        if itl:
            print(f"  ITL   mean={np.mean(itl):.2f}  P50={p(itl,50):.2f}  P90={p(itl,90):.2f}  ms")
        print(f"  TTFT >3s:  {100*sum(1 for x in ttft if x>3000)/n:.0f}%   >10s: {100*sum(1 for x in ttft if x>10000)/n:.0f}%")
    else:
        mttft = sorted([float(r["median_ttft_ms"]) for r in rows if r.get("median_ttft_ms")])
        mtps  = sorted([float(r["median_tps"])     for r in rows if r.get("median_tps")])
        n = len(mttft)
        p = lambda v, pct: v[int(pct*len(v)/100)]
        print(f"\n=== {label} median metrics (n={n} scenarios) ===")
        print(f"  median_TTFT  P50={p(mttft,50):.0f}  P90={p(mttft,90):.0f}  ms")
        print(f"  median_TPS   P50={p(mtps,50):.1f}   P90={p(mtps,90):.1f}")
EOF
```

### 4.2 Pandas deep-dive
```python
import pandas as pd
import numpy as np

EXP = "multi_domain_kv_thompson_9"  # ← change
BASE = f"examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/{EXP}/comparison"

llm = pd.read_csv(f"{BASE}/throughput_histogram_per_llm_call_data.csv")
req = pd.read_csv(f"{BASE}/throughput_histogram_data.csv")

# --- TTFT ---
print(llm["ttft_ms"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

# TTFT by call index (within session) — reveals within-session cascade
llm.groupby("llm_call_idx")["ttft_ms"].agg(
    ["median", lambda x: x.quantile(0.9), lambda x: (x > 10000).mean()]
).rename(columns={"<lambda_0>": "P90", "<lambda_1>": ">10s_rate"})

# Long-tail breakdown
for thresh in [1000, 3000, 5000, 10000]:
    pct = 100 * (llm["ttft_ms"] > thresh).mean()
    print(f"TTFT >{thresh/1000:.0f}s: {pct:.1f}%")

# --- TPS (throughput) ---
print(llm["tps"].describe(percentiles=[0.5, 0.9]))

# --- ITL ---
raw_itl = pd.read_csv(f"{BASE}/throughput_histogram_raw_itl_data.csv")
print(raw_itl["itl_ms"].describe(percentiles=[0.5, 0.9, 0.95]))

# --- Scenario-level: are bad calls isolated or clustered? ---
llm["is_high_ttft"] = llm["ttft_ms"] > 10000
bad_per_scenario = llm.groupby(["experiment", "example_number"])["is_high_ttft"].sum()
print("Scenarios with 1 bad call (isolated):", (bad_per_scenario == 1).sum())
print("Scenarios with 2+ bad calls (cascade):", (bad_per_scenario >= 2).sum())
```

### 4.3 Generate comparison plots (two experiments side-by-side)
```bash
SCRIPTS=examples/dynamo_integration/scripts
DEFAULT=examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_default_7
THOMPSON=examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_thompson_9
OUTPUT=/tmp/comparison

~/.venvs/nat_dynamo_eval/bin/python3 $SCRIPTS/plot_throughput_histograms_per_request.py \
    $DEFAULT $THOMPSON \
    --output $OUTPUT \
    --consolidate-legend

~/.venvs/nat_dynamo_eval/bin/python3 $SCRIPTS/plot_throughput_vs_tsq_per_request.py \
    $DEFAULT $THOMPSON \
    --output $OUTPUT \
    --consolidate-legend
```

---

## 5. Prometheus / Grafana

Grafana is at `http://localhost:3000` (no login required).

Key Prometheus queries to run at `http://localhost:9090`:

```promql
# TTFT percentiles from Dynamo frontend (does NOT include vLLM queue wait time)
histogram_quantile(0.95, rate(dynamo_frontend_time_to_first_token_seconds_bucket[5m]))
histogram_quantile(0.50, rate(dynamo_frontend_time_to_first_token_seconds_bucket[5m]))

# ITL
histogram_quantile(0.95, rate(dynamo_frontend_inter_token_latency_seconds_bucket[5m]))

# Thompson routing decisions per worker
rate(thompson_router_decisions_total[1m])

# KV opportunity cost — distribution of overlap delta (chosen vs best-idle)
# Positive = stayed on queued worker for better KV; negative = routed to idle at KV cost
histogram_quantile(0.50, rate(thompson_router_kv_vs_idle_delta_bucket[5m]))

# Routed-to-queued rate (decisions where idle worker existed but queued worker chosen)
rate(thompson_router_routed_to_queued_total[1m])

# KV overlap of the best-idle worker at decision time
histogram_quantile(0.50, rate(thompson_router_idle_kv_miss_bucket[5m]))

# NATS feedback latency (processor → router feedback RPC)
histogram_quantile(0.95, rate(thompson_router_feedback_latency_seconds_bucket[5m]))

# Worker queue depth at vLLM level (per worker)
dynamo_component_num_requests_waiting

# KV cache efficiency (Thompson processor counters)
rate(dynamo_component_thompson_kve_cached_tokens_total[1m]) /
rate(dynamo_component_thompson_kve_prompt_tokens_total[1m])
```

---

## 6. Cross-source Correlation: Diagnosing High TTFT

When you see elevated P95 TTFT, use this decision tree:

```
High P95 TTFT in profiler CSVs
│
├─ Check chosen_breakdown logs:
│    base≈0 for >50% decisions?
│    YES → load_mod_floor too low; increase load_mod_floor in config.yaml
│    NO  → proceed
│
├─ Check feedback elapsed_ms P90:
│    >15s?  Check consecutive logs:
│      consecutive > 5 common?  → within-session cascade (affinity lock-in)
│      consecutive ≤ 2 common?  → cross-session collision (random load spike)
│
├─ Check nats_overhead P90:
│    >100ms?  → NATS contention under load; consider collocating router+processor
│    <50ms?   → routing overhead is not the bottleneck
│
└─ Check kv_opportunity: QUEUED_PREFERRED rate:
     High rate + low extra_cold_tokens (<500)?
       → Router staying sticky when it shouldn't; reduce affinity_base
     High rate + high extra_cold_tokens (>5000)?
       → Router correctly paying prefill cost for KV efficiency
```

---

## 7. Log Locations Summary

| Log | Location | Notes |
|-----|----------|-------|
| Container stdout | `docker logs dynamo-vllm` | All components in one stream |
| Processor overhead | `external/dynamo/logs/processor_overhead.jsonl` | JSON Lines, one record per request |
| Profiler CSVs | `outputs/dynamo_evals/<exp>/comparison/*.csv` | Generated by `run_multi_domain_benchmark.sh` |
| Benchmark domain logs | `/tmp/multi_domain_*.log` | One file per domain, live during run |
| Grafana | `http://localhost:3000` | Historical metrics |
| Prometheus | `http://localhost:9090` | Raw queries |
