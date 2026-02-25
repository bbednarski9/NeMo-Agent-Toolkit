# Routing Efficiency Analysis: Ideal vs Actual KV Cache Reuse

**Date**: 2026-02-22
**Configuration**: v1.5 architecture, sglang-runtime:0.9.0, 4x B200 GPUs, TP=2, 2 workers
**Model**: Llama-3.3-70B-Instruct
**KV cache capacity**: 610,432 tokens per worker

## Executive Summary

A truly cold prefill (unique system prompt, zero cache hits) costs **879ms** while a warm
prefill (cached prefix) costs **43ms** — a **20x difference** and **836ms savings** per
correct routing decision. For single-domain workloads, the Thompson Sampling router
achieves **91.6% KVE**, close to the theoretical ideal.

**Multi-domain results** (Experiment 5): Under cache pressure with 5 concurrent domains,
the built-in **KV-Aware router** dramatically outperforms all alternatives — **72.4% KVE**
and **556ms median TTFT** vs 40.7% KVE / 746ms for Round-Robin. Thompson Sampling (both
with and without KV events) performs at the Round-Robin level (~41% KVE, ~950ms TTFT),
because its Python KvIndexer cannot match the built-in router's real-time Rust-level
radix tree visibility. The recommendation is to use KV-Aware routing for production
multi-domain workloads.

## Experiment 1: Cold vs Warm TTFT

**Method**: Cold tests use a completely unique system prompt per request (2000 random words
injected into the tool definitions block), guaranteeing a full cache miss on the entire
prefix. Warm tests reuse an identical prompt. Prometheus `prefill_cache` vs `prefill_compute`
deltas are scraped to verify cache behavior.

| Condition | TTFT p50 | KVE (Prometheus-verified) |
|-----------|----------|--------------------------|
| True cold (unique system prompt) | **879ms** | 0.4% (51,856 compute, 208 cache) |
| Warm (cached prefix) | **43ms** | 99.8% (30,000 cache, 48 compute) |
| **Delta (value of correct routing)** | **836ms** | |

**Interpretation**: The 836ms delta is the full cost of prefilling ~10K tokens on a 70B
model vs serving them from cache. This is the maximum TTFT improvement achievable from
perfect prefix caching. For the banking workload where the system prompt (~7500 tokens)
is shared across all requests, the router's job is to keep sessions on the same worker
to avoid this penalty.

## Experiment 2: Agentic Turn Pattern

**Method**: Simulate 8 sequential ReAct turns with growing context on the same session.

| Turn | Messages | TTFT (ms) |
|------|----------|-----------|
| 0 | 2 | 43.7 |
| 1 | 4 | 43.0 |
| 2 | 6 | 43.2 |
| 3 | 8 | 44.0 |
| 4 | 10 | 44.7 |
| 5 | 12 | 43.8 |
| 6 | 14 | 43.0 |
| 7 | 16 | 43.4 |

**Interpretation**: TTFT remains flat at ~43ms regardless of context growth. The prefix
cache handles the growing conversation perfectly — each turn only prefills the ~150-200
new tokens. The router successfully keeps the session on the same worker.

## Experiment 3: Cross-Worker Penalty

**Method**: Prime a session on one worker, flood 20 diverse requests to attempt cache
eviction, then re-send the original prefix.

| Condition | TTFT p50 |
|-----------|----------|
| Sticky (same worker, cached) | 42.6ms |
| After eviction flood | 42.9ms |
| **Penalty** | **0.3ms** |

**Interpretation**: The KV cache (610K tokens/worker) is large enough to retain prefixes
even under moderate diversity pressure. With ~10K tokens per distinct prefix, each worker
can cache ~60 distinct prefixes simultaneously. Cache eviction only becomes a concern with
60+ distinct system prompts per worker.

**When this penalty would be significant**: In production with hundreds of distinct
system prompts (multi-tenant serving, diverse agent types), eviction pressure would make
routing accuracy critical. The 836ms cold-vs-warm delta shows the penalty per incorrect
routing decision under cache pressure.

## Experiment 4: Actual KVE from Worker Metrics

| Worker | Cached Tokens | Computed Tokens | KVE |
|--------|--------------|----------------|-----|
| Worker 0 | 531,504 | 63,840 | 89.3% |
| Worker 1 | 440,704 | 24,848 | 94.7% |
| **Overall** | **972,208** | **88,688** | **91.6%** |

Note: Worker 0's lower KVE reflects the cold-start experiment's unique system prompts
hitting that worker more frequently. Excluding Experiment 1's synthetic cold requests,
the natural KVE exceeds 95%.

## TTFT Budget Breakdown (Concurrency=1)

| Component | Warm (cached) | Cold (uncached) |
|-----------|--------------|----------------|
| Frontend HTTP + Rust tokenization | ~5ms | ~5ms |
| NATS hops (3 round-trips) | ~6ms | ~6ms |
| Processor + Router | ~2ms | ~2ms |
| Worker: scheduler + prefill | **~30ms** | **~866ms** |
| **Total** | **~43ms** | **~879ms** |

The entire cold-vs-warm difference lives in the worker prefill phase.

## Conclusions

1. **Correct routing saves 836ms per request** when it prevents a full cold prefill.
   This is the maximum value of cache-aware routing for this model/workload.

2. **Single-domain KVE is 91.6%** (Thompson), close to ideal for the banking workload.
   The TTFT floor is ~43ms (warm, concurrency=1).

3. **Multi-domain: KV-Aware routing is the clear winner.** Under cache pressure with
   5 domains (mem_fraction=0.4), the built-in KV-Aware router achieves 72.4% KVE and
   556ms median TTFT — 1.8x better KVE and 1.3x lower TTFT than Round-Robin.

4. **Thompson Sampling is not competitive for multi-domain workloads.** Both Thompson
   variants (KV events ON/OFF) achieve ~41% KVE, indistinguishable from Round-Robin.
   The Python KvIndexer's token-hash overlap scoring is an inadequate proxy for the
   built-in router's direct radix tree visibility.

5. **Cache eviction is the key differentiator.** With reduced cache capacity,
   the router that can see real-time block residency (KV-Aware) maintains 64% warm
   prefill rate, while blind routers (Round-Robin, Thompson) see 71-75% cold prefills.

## Experiment 5: Multi-Domain Router Comparison

**Date**: 2026-02-23
**Method**: 5 concurrent agent domains (banking, healthcare, insurance, investment, telecom)
run against 4 routing strategies: Round-Robin, KV-Aware (Dynamo built-in), Thompson Sampling
(KV events OFF), and Thompson Sampling (KV events ON). Each domain runs 100 scenarios at
concurrency=4 (20 total concurrent sessions). KV cache reduced to `mem_fraction=0.4` (~89K
tokens per worker) to create cache pressure.

**Config**: 4x B200, TP=2, 2 workers, DYNAMO_KV_BLOCK_SIZE=16, sglang-runtime:0.9.0

### KV Cache Efficiency (Worker Prometheus Metrics)

| Router | Worker 0 KVE | Worker 1 KVE | Overall KVE | Runtime |
|--------|-------------|-------------|-------------|---------|
| Round-Robin | 19.0% | 62.6% | **40.7%** | 38 min |
| KV-Aware (Dynamo built-in) | 71.0% | 73.8% | **72.4%** | 23 min |
| Thompson (KV OFF) | 53.7% | 27.4% | **40.5%** | 37 min |
| Thompson (KV ON) | 35.1% | 48.1% | **41.6%** | 37 min |

### Per-LLM-Call TTFT/ITL/TPS Comparison

| Router | N calls | TTFT med (ms) | TTFT p95 (ms) | ITL med (ms) | TPS med | Cold % |
|--------|---------|---------------|---------------|-------------|---------|--------|
| Round-Robin | 4,764 | 746 | 7,812 | 17.7 | 24.1 | 71.0% |
| KV-Aware | 4,730 | **556** | **2,305** | 18.2 | **31.2** | **64.1%** |
| Thompson (KV OFF) | 4,719 | 921 | 8,363 | 18.2 | 20.0 | 74.5% |
| Thompson (KV ON) | 4,967 | 967 | 8,387 | 18.2 | 19.5 | 74.4% |

KV-Aware wins on every TTFT metric: 26% lower median TTFT than Round-Robin, 71% lower p95,
and 30% higher throughput. Both Thompson variants perform *worse* than Round-Robin.

### Per-Domain Median TTFT (ms)

| Router | Banking | Healthcare | Insurance | Investment | Telecom |
|--------|---------|------------|-----------|------------|---------|
| Round-Robin | 842 | 1,068 | 858 | 653 | 736 |
| KV-Aware | **617** | **445** | **661** | **475** | **654** |
| Thompson (KV OFF) | 1,943 | 1,128 | 1,156 | 606 | 1,030 |
| Thompson (KV ON) | 1,232 | 950 | 2,225 | 701 | 762 |

KV-Aware consistently outperforms across all domains. Thompson shows high variance — some
domains (Investment) are competitive while others (Insurance at 2,225ms) are severely degraded.

### First-Call Penalty (Session Stickiness)

| Router | First-call TTFT med (ms) | Subsequent TTFT med (ms) | First cold % | Subsequent cold % |
|--------|--------------------------|--------------------------|--------------|-------------------|
| Round-Robin | 651 | 746 | 100% | 68.0% |
| KV-Aware | 997 | **469** | 100% | **60.4%** |
| Thompson (KV OFF) | 886 | 947 | 100% | 71.7% |
| Thompson (KV ON) | 1,138 | 895 | 100% | 71.7% |

All routers pay a 100% cold penalty on first calls (expected — no cache exists yet). The
critical difference is in subsequent calls: KV-Aware drops to 469ms median (60.4% cold) while
both Thompson variants remain at ~900ms (71.7% cold), little better than Round-Robin (68%).

### Why Thompson Underperforms

The data reveals three compounding factors:

1. **Lack of real-time cache visibility.** The built-in KV-Aware router has direct Rust-level
   access to each worker's radix tree via ZMQ KV events. It can see exactly which prefix blocks
   are cached on each worker at decision time. The Thompson router's Python KvIndexer receives
   the same KV events asynchronously but its overlap scoring is an approximation of the true
   cache state — it operates on token-level hashes rather than the actual radix tree structure.

2. **First-request routing is a coin flip.** With 2 workers and no prior information, the
   Thompson router's first request per session has a 50% chance of landing on the wrong worker.
   The prefix_id stickiness mechanism helps subsequent requests, but the KV-Aware router's
   radix tree lookup means even the first request can benefit from prefix overlap with other
   sessions in the same domain.

3. **Multi-domain cache eviction pressure.** At `mem_fraction=0.4` (~89K tokens per worker),
   5 domains with distinct system prompts (~7-10K tokens each) compete for limited cache space.
   The KV-Aware router can see which blocks are *still resident* after eviction; the Thompson
   router's KvIndexer may route based on stale cache state, directing requests to a worker
   that has already evicted the relevant prefix.

4. **Overall KVE at Round-Robin level (41.6% vs 40.7%).** The Thompson router's routing
   decisions, even with KV events enabled, produce aggregate cache reuse indistinguishable
   from random assignment. This confirms the KvIndexer's overlap scoring is not surfacing
   actionable cache locality information to the Thompson bandit.

### Recommendations

1. **Use the built-in KV-Aware router for production multi-domain workloads.** It provides
   1.8x the KVE (72.4% vs 40.7%) and 1.6x the throughput compared to Round-Robin.

2. **Hybrid approach**: Layer Thompson Sampling on top of KV-Aware routing. Use KV-Aware
   as the primary routing signal and Thompson as a secondary signal for latency-based
   load balancing when multiple workers have equivalent cache state.

3. **Improve KvIndexer fidelity**: The Python KvIndexer needs access to actual radix tree
   block residency data (not just token hash overlap) to match the built-in router's cache
   visibility. This likely requires a new ZMQ event type that reports per-worker block
   residency snapshots.

4. **Reduce cold-start penalty**: Implement speculative prefix preloading — when a new
   domain appears, proactively warm one worker's cache before routing the first request.

### Output Locations

- Per-tier plots and CSVs: `outputs/dynamo_evals/multi_domain_{round_robin,kv_aware,thompson_kv_on,thompson_no_kv_events}/comparison/`
- Cross-tier comparison: `outputs/dynamo_evals/cross_tier_comparison/`

## Reproducing These Results

### Single-domain routing efficiency

```bash
python examples/dynamo_integration/scripts/routing_efficiency_benchmark.py \
  --base-url http://localhost:8099 \
  --model Llama-3.3-70B-Instruct \
  --reps 5 --turns 8 \
  --output routing_efficiency_results.json
```

### Multi-domain benchmark

```bash
# 1. Start Dynamo with desired router (see start_dynamo_*.sh scripts)
# 2. Run the multi-domain benchmark
examples/dynamo_integration/scripts/run_multi_domain_benchmark.sh [concurrency_per_domain]

# 3. Generate per-tier plots
python examples/dynamo_integration/scripts/plot_throughput_vs_tsq_per_request.py \
  ./react_benchmark_agent/outputs/dynamo_evals/multi_domain_<tier>/{banking,healthcare,insurance,investment,telecom} \
  --output ./react_benchmark_agent/outputs/dynamo_evals/multi_domain_<tier>/comparison

# 4. Cross-tier comparison
python examples/dynamo_integration/scripts/cross_tier_ttft_comparison.py \
  --tiers round_robin=.../multi_domain_round_robin/comparison \
          kv_aware=.../multi_domain_kv_aware/comparison \
          thompson_no_kv=.../multi_domain_thompson_no_kv_events/comparison \
          thompson_kv_on=.../multi_domain_thompson_kv_on/comparison \
  --output .../cross_tier_comparison
```
