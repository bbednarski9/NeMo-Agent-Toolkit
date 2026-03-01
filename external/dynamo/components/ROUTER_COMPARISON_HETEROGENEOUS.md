# Router Comparison: Heterogeneous KV Cache Proxy

Benchmark comparing native KV-aware routing vs Thompson Sampling routing
on a heterogeneous 8-worker proxy with KV cache sizes ranging from 3,000
to 10,000 blocks per worker.

## Setup

- **Model:** Llama-3.1-8B-Instruct (vLLM, unified mode)
- **Workers:** 8 (TP=1 each, GPUs 0-7)
- **KV Cache (heterogeneous):** 3K, 4K, 5K, 6K, 7K, 8K, 9K, 10K blocks (block_size=16)
- **Benchmark:** 5 domains x 5 variants, 25 prompt configs, concurrency 4 per config
- **Dataset:** Agent Leaderboard v2 (multi-domain, multi-turn agentic tasks)
- **Image:** `dynamo-vllm-source:main` (built from Dynamo main branch)

## Results

### Head-to-Head Summary

| Metric | kv_native_1 | kv_thompson_native_1 | Delta |
|--------|-------------|---------------------|-------|
| **n (LLM calls)** | 698 | 720 | +22 |
| **TTFT P50** | 409ms | **97ms** | **-312ms (4.2x faster)** |
| **TTFT P90** | **652ms** | 683ms | +31ms |
| **TTFT P95** | **806ms** | 941ms | +135ms |
| **TTFT P99** | 1420ms | 1613ms | +193ms |
| **TTFT max** | 2513ms | 2775ms | +262ms |
| **TTFT >1s** | 3.7% | 4.4% | +0.7% |
| **TTFT >3s** | 0.0% | 0.0% | tie |
| **TPS P50** | 47.0 | **63.7** | **+16.7 (+35%)** |
| **TPS mean** | 53.4 | **70.1** | **+16.7 (+31%)** |
| **TPS P10** | 32.2 | 34.9 | +2.7 |
| **TPS P90** | 81.0 | **118.7** | **+37.7 (+47%)** |
| **ITL P50** | 9.05ms | **8.59ms** | **-0.46ms (-5%)** |
| **ITL P90** | **10.06ms** | 10.85ms | +0.79ms |
| **ITL P95** | **10.30ms** | 11.73ms | +1.43ms |
| **ITL mean** | 8.93ms | **8.75ms** | **-0.18ms (-2%)** |
| Routing overhead P50 | 4.0ms | 3.5ms | -0.5ms |

### Per-Domain Breakdown

| Domain | n (native) | TTFT P50 (native) | n (thompson) | TTFT P50 (thompson) | TPS P50 (native) | TPS P50 (thompson) |
|--------|-----------|-------------------|-------------|---------------------|------------------|---------------------|
| banking | 58 | 453ms | 79 | **344ms** | 34.7 | **42.9** |
| healthcare | 102 | 385ms | 77 | **305ms** | 49.2 | **58.9** |
| insurance | 205 | 396ms | 157 | **92ms** | 46.8 | **63.8** |
| investment | 161 | 407ms | 209 | **91ms** | 54.0 | **82.2** |
| telecom | 172 | 412ms | 198 | **331ms** | 46.7 | **59.9** |

### TTFT by LLM Call Index (within session)

| Call | kv_native P50 | kv_native P90 | thompson P50 | thompson P90 |
|------|--------------|--------------|-------------|-------------|
| 0 (cold) | 427ms | 1085ms | 431ms | 913ms |
| 1 | 364ms | 711ms | **100ms** | 731ms |
| 2 | 432ms | 478ms | **85ms** | 592ms |
| 3 | 372ms | 596ms | **89ms** | 701ms |
| 4 | 391ms | 701ms | **87ms** | 489ms |
| 5 | 364ms | 541ms | **90ms** | 531ms |
| 6 | 377ms | 500ms | **87ms** | 543ms |
| 7 | 419ms | 502ms | **87ms** | 512ms |

## Learner Analysis (Thompson Router)

### Routing Decisions
- 5,569 total decisions
- **Overrode native router 69.9%** of the time (agreed 30.1%)
- Agreement increased over time: 21.2% early -> 33.8% late (learner converging)

### Session Stickiness (key to 4x TTFT improvement)
- **65.6% of follow-up calls stayed on the same worker** (KV cache reuse)
- Average 3.6 unique workers per session (out of 8)
- 93% of multi-call sessions used 2+ workers (healthy exploration)
- Only 7% fully locked to one worker

### Worker Load Distribution
- Balanced: 10.6%-14.5% per worker (12.5% = perfectly uniform)
- At concurrency 4, even the 3K-block worker wasn't bottlenecking

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full architecture diagrams comparing
the original NATS-based Thompson router with the new in-process KvRouter integration.

## Key Findings

1. **Session stickiness is the dominant factor.** The native router evaluates every
   request independently; Thompson's affinity feature learns to keep sessions on the
   same worker for KV cache reuse, dropping TTFT from 409ms to 97ms for follow-up calls.

2. **The P90/P95 tail is slightly worse with Thompson** (+31ms at P90, +135ms at P95)
   due to exploration cost — the learner occasionally tries suboptimal workers to gather
   information. This is tunable via the exploration weight.

3. **Both routers achieve 0% >3s TTFT** thanks to the native `ActiveSequences`
   lifecycle management (proper in-flight tracking, mark_prefill_complete, free).

4. **Thompson's TPS improvement (+35%)** comes from reduced prefill overhead when
   sessions are sticky — less redundant KV cache computation.

## Raw Data

- Native baseline: `examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_native_1/`
- Thompson native: `examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_kv_thompson_native_1/`
- Per-LLM-call CSVs: `comparison/throughput_histogram_per_llm_call_data.csv`
- Docker logs: `docker logs dynamo-vllm` (routing decisions, feedback, learner updates)
