# Router Comparison: Homogeneous KV Cache Proxy

Benchmark comparing native KV-aware routing vs Thompson Sampling routing
on a homogeneous 8-worker proxy where all workers have identical KV cache
capacity (6,500 blocks each — the mean of the heterogeneous 3K-10K setup).

## Setup

- **Model:** Llama-3.1-8B-Instruct (vLLM, unified mode)
- **Workers:** 8 (TP=1 each, GPUs 0-7)
- **KV Cache (homogeneous):** 6,500 blocks per worker (block_size=16)
- **Total cluster KV capacity:** 52,000 blocks (same as heterogeneous mean)
- **Benchmark:** 5 domains x 5 variants, 25 prompt configs, concurrency 4 per config
- **Dataset:** Agent Leaderboard v2 (multi-domain, multi-turn agentic tasks)
- **Image:** `dynamo-vllm-source:main` (built from Dynamo main branch)

## Results

### Head-to-Head Summary

| Metric | kv_native_2 | kv_thompson_native_2 | Delta |
|--------|-------------|---------------------|-------|
| **n (LLM calls)** | 743 | 727 | -16 |
| **TTFT P50** | 383ms | **87ms** | **-296ms (4.4x faster)** |
| **TTFT P90** | 536ms | **497ms** | **-39ms** |
| **TTFT P95** | 687ms | **615ms** | **-72ms** |
| **TTFT P99** | 970ms | **1256ms** | +286ms |
| **TTFT max** | 2354ms | 2502ms | +148ms |
| **TTFT >1s** | **0.9%** | 2.1% | +1.2% |
| **TTFT >3s** | 0.0% | 0.0% | tie |
| **TPS P50** | 50.7 | **74.5** | **+23.8 (+47%)** |
| **TPS mean** | 59.8 | **80.4** | **+20.6 (+34%)** |
| **TPS P10** | 33.4 | **43.1** | **+9.7 (+29%)** |
| **TPS P90** | 103.6 | **125.1** | **+21.5 (+21%)** |
| **ITL P50** | 9.26ms | **8.63ms** | **-0.63ms (-7%)** |
| **ITL P90** | **9.96ms** | 10.63ms | +0.67ms |
| **ITL P95** | **10.14ms** | 11.40ms | +1.26ms |
| **ITL mean** | 8.88ms | **8.69ms** | **-0.19ms (-2%)** |

### Per-Domain Breakdown

| Domain | n (native) | TTFT P50 (native) | n (thompson) | TTFT P50 (thompson) | TPS P50 (native) | TPS P50 (thompson) |
|--------|-----------|-------------------|-------------|---------------------|------------------|---------------------|
| banking | 60 | 399ms | 88 | **89ms** | 35.2 | **63.8** |
| healthcare | 114 | 357ms | 108 | **86ms** | 53.0 | **93.4** |
| insurance | 205 | 415ms | 177 | **86ms** | 48.7 | **73.7** |
| investment | 183 | 344ms | 181 | **87ms** | 62.0 | **87.2** |
| telecom | 181 | 413ms | 173 | **87ms** | 48.6 | **67.3** |

### TTFT by LLM Call Index (within session)

| Call | kv_native P50 | kv_native P90 | thompson P50 | thompson P90 |
|------|--------------|--------------|-------------|-------------|
| 0 (cold) | 421ms | 741ms | 439ms | 986ms |
| 1 | 341ms | 459ms | **79ms** | 507ms |
| 2 | 349ms | 464ms | **82ms** | 462ms |
| 3 | 354ms | 473ms | **82ms** | 457ms |
| 4 | 373ms | 481ms | **82ms** | 461ms |
| 5 | 437ms | 486ms | **85ms** | 480ms |
| 6 | 396ms | 516ms | **86ms** | 496ms |
| 7 | 433ms | 499ms | **84ms** | 401ms |

### TTFT Distribution

| Range | kv_native_2 | kv_thompson_native_2 |
|-------|------------|---------------------|
| 0-200ms | 265 (35.7%) | **487 (67.0%)** |
| 200-500ms | 378 (50.9%) | 170 (23.4%) |
| 500-1000ms | 93 (12.5%) | 55 (7.6%) |
| 1000-2000ms | 5 (0.7%) | 12 (1.7%) |
| 2000-5000ms | 2 (0.3%) | 3 (0.4%) |
| >5000ms | 0 (0.0%) | 0 (0.0%) |

## Key Findings

1. **Thompson's advantage is even larger on homogeneous hardware.** TTFT P50 improved
   from 383ms to 87ms (4.4x) — slightly better than the 4.2x on heterogeneous. With
   uniform cache sizes, the native router has no worker differentiation at all; Thompson's
   session stickiness becomes the only signal that matters.

2. **Thompson wins across the entire TPS distribution.** P10 improved +29%, P50 +47%,
   P90 +21%. On homogeneous hardware, all workers are equally capable, so Thompson's
   ability to reuse KV cache (reducing prefill) translates directly to higher throughput.

3. **Per-domain TTFT is remarkably uniform with Thompson** — all 5 domains land at
   86-89ms P50. On the native router, domains ranged from 344ms to 415ms. Thompson
   eliminates domain-dependent variation by learning session-level stickiness.

4. **Native router's tail is tighter on homogeneous hardware.** P99=970ms (native) vs
   1256ms (Thompson). Without the 3K-block bottleneck worker, the native router's
   uniform spread avoids worst-case scenarios. Thompson's exploration cost shows up
   more at the P99 level.

5. **Both routers maintain 0% >3s TTFT** — the native KvRouter lifecycle management
   keeps the system stable under load.

## Comparison: Heterogeneous vs Homogeneous

| Metric | Heterogeneous (run 1) | Homogeneous (run 2) | Notes |
|--------|--------------------|--------------------|----|
| Native TTFT P50 | 409ms | **383ms** | Homo slightly better (no 3K bottleneck) |
| Native TTFT P90 | 652ms | **536ms** | Homo much better tail |
| Thompson TTFT P50 | 97ms | **87ms** | Homo slightly better |
| Thompson TTFT P90 | 683ms | **497ms** | Homo much better tail |
| Thompson TPS P50 | 63.7 | **74.5** | Homo +17% TPS |
| Thompson advantage (P50) | 4.2x | **4.4x** | Larger on homogeneous |

## Raw Data

- Native baseline: `examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_native_2/`
- Thompson native: `examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_thompson_native_2/`
- Per-LLM-call CSVs: `comparison/throughput_histogram_per_llm_call_data.csv`
