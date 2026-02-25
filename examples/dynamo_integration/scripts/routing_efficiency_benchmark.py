#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Routing Efficiency Benchmark: Ideal vs Actual KV Cache Reuse.

Measures the TTFT ceiling achievable with perfect prefix caching and compares
it to actual router performance to quantify room for improvement.

Experiments:
  1. Cold vs Warm TTFT — value of a correct routing decision
  2. Agentic Turn Pattern — TTFT trajectory as context grows per turn
  3. Cross-Worker Penalty — cost of routing to the wrong worker

Usage:
    python routing_efficiency_benchmark.py [--base-url http://localhost:8099]
                                           [--model Llama-3.3-70B-Instruct]
                                           [--worker-metrics-port 18081]
                                           [--num-workers 2]
                                           [--turns 10]
                                           [--reps 5]
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scrape_worker_prefill(base_port: int, num_workers: int) -> dict[int, dict[str, float]]:
    """Scrape prefill_cache and prefill_compute from each worker's Prometheus."""
    result = {}
    for i in range(num_workers):
        port = base_port + i
        cache = 0.0
        compute = 0.0
        try:
            body = urlopen(f"http://localhost:{port}/metrics", timeout=3).read().decode()
            for line in body.splitlines():
                if line.startswith("#"):
                    continue
                if 'mode="prefill_cache"' in line and "realtime_tokens_total" in line:
                    cache = float(line.rsplit(" ", 1)[-1])
                elif 'mode="prefill_compute"' in line and "realtime_tokens_total" in line:
                    compute = float(line.rsplit(" ", 1)[-1])
        except Exception:
            pass
        result[i] = {"cache": cache, "compute": compute}
    return result


def _ttft_streaming(url: str, payload: dict, skip_empty: bool = True) -> tuple[float, int]:
    """Send a streaming chat completion and return (ttft_ms, prompt_tokens).

    Uses the corrected TTFT method: skips the initial empty SSE chunk that
    streaming APIs emit before the real first token.
    """
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=120)

    t_start = time.perf_counter()
    first_real_token_time = None
    prompt_tokens = 0
    got_empty_first = False

    for raw_line in resp:
        line = raw_line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue

        chunk_json = line[6:]
        try:
            chunk = json.loads(chunk_json)
        except json.JSONDecodeError:
            continue

        # Extract prompt_tokens from the final usage chunk
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage.get("prompt_tokens"):
            prompt_tokens = usage["prompt_tokens"]

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if first_real_token_time is None:
            if not content and skip_empty and not got_empty_first:
                got_empty_first = True
                continue
            first_real_token_time = time.perf_counter()

    resp.close()
    if first_real_token_time is None:
        first_real_token_time = time.perf_counter()

    ttft_ms = (first_real_token_time - t_start) * 1000
    return ttft_ms, prompt_tokens


def _build_system_prompt(unique: bool = False) -> str:
    """Build a realistic system prompt matching the banking ReAct workload.

    Args:
        unique: If True, inject a large random block into the prompt so
                the entire prefix is guaranteed to miss the KV cache.
    """
    tools_path = Path(__file__).parent.parent / "data" / "raw" / "banking" / "tools.json"
    if tools_path.exists():
        tools_json = json.loads(tools_path.read_text())
        tools_str = json.dumps(tools_json, indent=2)
    else:
        tools_str = "[tool definitions not found — using placeholder]" * 100

    salt = ""
    if unique:
        # ~2000 random words ≈ 2500 tokens — large enough to guarantee a
        # complete cache miss on the entire prefix, not just a suffix.
        import random
        words = ["alpha", "bravo", "cache", "delta", "echo", "foxtrot",
                 "gamma", "hotel", "india", "juliet", "kilo", "lima"]
        salt = "\n\n[BENCHMARK SALT — IGNORE]\n" + " ".join(
            random.choice(words) for _ in range(2000)
        ) + "\n[END SALT]\n"

    return (
        "You are a tool-calling agent evaluated on TOOL SELECTION capability. "
        "Your goal is to select ALL the correct tools, in the correct order.\n\n"
        f"{salt}"
        "Available tools:\n\n"
        f"{tools_str}\n\n"
        "Use this exact format for EACH response:\n"
        "Thought: ...\nAction: ...\nAction Input: ...\n"
    )


def _prefill_deltas(before: dict, after: dict) -> dict:
    """Compute per-worker prefill_cache and prefill_compute deltas."""
    result = {}
    for wid in after:
        dc = after[wid]["cache"] - before.get(wid, {}).get("cache", 0)
        dd = after[wid]["compute"] - before.get(wid, {}).get("compute", 0)
        total = dc + dd
        kve = dc / total * 100 if total > 0 else 0
        result[wid] = {"cache_delta": dc, "compute_delta": dd, "kve": kve}
    return result


# ---------------------------------------------------------------------------
# Experiment 1: Cold vs Warm TTFT
# ---------------------------------------------------------------------------

def experiment_cold_vs_warm(base_url: str, model: str, reps: int,
                           worker_metrics_port: int = 18081, num_workers: int = 2):
    """Measure TTFT for truly cold (unique system prompt) vs warm (cached)."""
    print("=" * 60)
    print("Experiment 1: Cold vs Warm TTFT")
    print("=" * 60)

    system_prompt = _build_system_prompt(unique=False)
    url = f"{base_url}/v1/chat/completions"
    user_query = "I need to check my account balance and recent transactions."

    warm_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 50,
        "stream": True,
    }

    # Prime the cache with one request
    print("  Priming cache with standard system prompt...")
    _ttft_streaming(url, warm_payload)
    time.sleep(1)

    # --- Warm TTFT (identical request, prefix fully cached) ---
    before = _scrape_worker_prefill(worker_metrics_port, num_workers)
    warm_ttfts = []
    print(f"  Measuring warm TTFT ({reps} reps, identical prompt)...")
    for i in range(reps):
        ttft, pt = _ttft_streaming(url, warm_payload)
        warm_ttfts.append(ttft)
        print(f"    [{i}] TTFT={ttft:.1f}ms  prompt_tokens={pt}")
        time.sleep(0.5)
    after = _scrape_worker_prefill(worker_metrics_port, num_workers)
    warm_deltas = _prefill_deltas(before, after)
    print(f"  Warm prefill breakdown: {warm_deltas}")

    # --- Cold TTFT (entirely unique system prompt per request) ---
    before = _scrape_worker_prefill(worker_metrics_port, num_workers)
    cold_ttfts = []
    print(f"  Measuring cold TTFT ({reps} reps, UNIQUE system prompt each)...")
    for i in range(reps):
        cold_system = _build_system_prompt(unique=True)
        cold_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": cold_system},
                {"role": "user", "content": user_query},
            ],
            "max_tokens": 50,
            "stream": True,
        }
        ttft, pt = _ttft_streaming(url, cold_payload)
        cold_ttfts.append(ttft)
        print(f"    [{i}] TTFT={ttft:.1f}ms  prompt_tokens={pt}")
        time.sleep(0.5)
    after = _scrape_worker_prefill(worker_metrics_port, num_workers)
    cold_deltas = _prefill_deltas(before, after)
    print(f"  Cold prefill breakdown: {cold_deltas}")

    cold_ttfts.sort()
    warm_ttfts.sort()
    cn, wn = len(cold_ttfts), len(warm_ttfts)

    cold_p50 = cold_ttfts[cn // 2]
    warm_p50 = warm_ttfts[wn // 2]
    delta = cold_p50 - warm_p50

    print()
    print("  Results:")
    print(f"    Cold TTFT p50:  {cold_p50:.1f}ms  (mean={sum(cold_ttfts)/cn:.1f}ms)")
    print(f"    Warm TTFT p50:  {warm_p50:.1f}ms  (mean={sum(warm_ttfts)/wn:.1f}ms)")
    print(f"    Delta (value of correct routing): {delta:.1f}ms")
    print()

    return {"cold_p50": cold_p50, "warm_p50": warm_p50, "delta": delta,
            "cold_all": cold_ttfts, "warm_all": warm_ttfts,
            "cold_prefill_deltas": cold_deltas, "warm_prefill_deltas": warm_deltas}


# ---------------------------------------------------------------------------
# Experiment 2: Agentic Turn Pattern
# ---------------------------------------------------------------------------

def experiment_agentic_turns(base_url: str, model: str, num_turns: int):
    """Simulate a ReAct session with growing context, measuring TTFT per turn."""
    print("=" * 60)
    print(f"Experiment 2: Agentic Turn Pattern ({num_turns} turns)")
    print("=" * 60)

    system_prompt = _build_system_prompt()
    url = f"{base_url}/v1/chat/completions"

    user_query = "I need to check my account balance, review recent transactions, and transfer $500 to savings."

    # Simulated tool responses (~150-200 tokens each)
    mock_tool_responses = [
        "Thought: I need to check the account balance first.\nAction: get_account_balance\nAction Input: {\"account_id\": \"12345\"}",
        "Observation: {\"balance\": 5432.10, \"currency\": \"USD\", \"account_type\": \"checking\"}",
        "Thought: Now I need to get the transaction history.\nAction: get_transaction_history\nAction Input: {\"account_id\": \"12345\", \"start_date\": \"2024-01-01\"}",
        "Observation: {\"transactions\": [{\"date\": \"2024-01-15\", \"amount\": -50.00, \"description\": \"Coffee Shop\"}, {\"date\": \"2024-01-14\", \"amount\": -120.00, \"description\": \"Grocery Store\"}]}",
        "Thought: I need to transfer funds to savings.\nAction: transfer_funds\nAction Input: {\"from_account\": \"12345\", \"to_account\": \"67890\", \"amount\": 500}",
        "Observation: {\"transfer_id\": \"TXN-001\", \"status\": \"completed\", \"new_balance\": 4932.10}",
        "Thought: Let me verify the transfer was successful.\nAction: get_account_balance\nAction Input: {\"account_id\": \"12345\"}",
        "Observation: {\"balance\": 4932.10, \"currency\": \"USD\", \"account_type\": \"checking\"}",
        "Thought: Let me also check the savings account balance.\nAction: get_account_balance\nAction Input: {\"account_id\": \"67890\"}",
        "Observation: {\"balance\": 10500.00, \"currency\": \"USD\", \"account_type\": \"savings\"}",
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    turn_results = []

    for turn in range(num_turns):
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": 100,
            "stream": True,
        }

        ttft, pt = _ttft_streaming(url, payload)
        turn_results.append({"turn": turn, "ttft_ms": ttft, "prompt_tokens": pt, "num_messages": len(messages)})
        print(f"  Turn {turn:2d}: TTFT={ttft:7.1f}ms  prompt_tokens={pt:5d}  messages={len(messages)}")

        # Grow context with simulated assistant + tool responses
        if turn < len(mock_tool_responses):
            messages.append({"role": "assistant", "content": mock_tool_responses[turn]})
        else:
            messages.append({"role": "assistant", "content": f"Thought: Continuing analysis step {turn}.\nAction: get_account_balance\nAction Input: {{\"account_id\": \"12345\"}}"})

        if turn + 1 < num_turns:
            messages.append({"role": "user", "content": f"Observation: {{\"status\": \"ok\", \"step\": {turn}}}"})

        time.sleep(0.3)

    # Compute cache efficiency from turn deltas
    if len(turn_results) >= 2 and turn_results[0]["prompt_tokens"] > 0:
        first_pt = turn_results[0]["prompt_tokens"]
        last_pt = turn_results[-1]["prompt_tokens"]
        total_new = sum(
            max(0, turn_results[i]["prompt_tokens"] - turn_results[i - 1]["prompt_tokens"])
            for i in range(1, len(turn_results))
        )
        total_prompt = sum(r["prompt_tokens"] for r in turn_results)
        ideal_cached = total_prompt - first_pt - total_new
        ideal_kve = ideal_cached / total_prompt * 100 if total_prompt > 0 else 0

        print()
        print(f"  Ideal KVE for this session: {ideal_kve:.1f}%")
        print(f"  Total prompt tokens across all turns: {total_prompt:,}")
        print(f"  Tokens that COULD be cached: {ideal_cached:,}")
        print(f"  New tokens per turn (avg): {total_new / max(1, len(turn_results) - 1):.0f}")

    print()
    return turn_results


# ---------------------------------------------------------------------------
# Experiment 3: Cross-Worker Penalty
# ---------------------------------------------------------------------------

def experiment_cross_worker_penalty(base_url: str, model: str, reps: int):
    """Measure the TTFT penalty when a session switches workers.

    Since we can't force a specific worker via the external API, we use a
    proxy: send the same prompt back-to-back rapidly (likely hits the same
    worker) vs send it after a burst of different prompts that may shift
    the router's decision.
    """
    print("=" * 60)
    print("Experiment 3: Cross-Worker Penalty Estimation")
    print("=" * 60)

    system_prompt = _build_system_prompt()
    url = f"{base_url}/v1/chat/completions"

    session_query = "I need help with my mortgage payment options and refinancing."

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": session_query},
    ]

    # Prime the cache
    print("  Priming session on a worker...")
    for _ in range(3):
        payload = {"model": model, "messages": base_messages, "max_tokens": 50, "stream": True}
        _ttft_streaming(url, payload)
        time.sleep(0.3)

    # Measure sticky (same prefix, likely same worker)
    sticky_ttfts = []
    print(f"  Measuring sticky routing ({reps} reps)...")
    for i in range(reps):
        payload = {"model": model, "messages": base_messages, "max_tokens": 50, "stream": True}
        ttft, pt = _ttft_streaming(url, payload)
        sticky_ttfts.append(ttft)
        print(f"    [{i}] TTFT={ttft:.1f}ms")
        time.sleep(0.3)

    # Now flood with diverse requests to pollute both workers' caches,
    # then re-send the original prefix
    print("  Flooding with diverse requests to evict cache...")
    for i in range(20):
        diverse_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Unique diversionary request number {uuid.uuid4().hex} about topic {i}"},
            ],
            "max_tokens": 10,
            "stream": True,
        }
        _ttft_streaming(url, diverse_payload)

    # Measure post-eviction (prefix likely evicted from cache)
    evicted_ttfts = []
    print(f"  Measuring post-eviction TTFT ({reps} reps)...")
    for i in range(reps):
        payload = {"model": model, "messages": base_messages, "max_tokens": 50, "stream": True}
        ttft, pt = _ttft_streaming(url, payload)
        evicted_ttfts.append(ttft)
        print(f"    [{i}] TTFT={ttft:.1f}ms")
        time.sleep(0.3)

    sticky_ttfts.sort()
    evicted_ttfts.sort()
    sn, en = len(sticky_ttfts), len(evicted_ttfts)

    sticky_p50 = sticky_ttfts[sn // 2]
    evicted_p50 = evicted_ttfts[en // 2]
    penalty = evicted_p50 - sticky_p50

    print()
    print("  Results:")
    print(f"    Sticky (cached) p50:   {sticky_p50:.1f}ms")
    print(f"    Evicted (cold) p50:    {evicted_p50:.1f}ms")
    print(f"    Penalty (eviction cost): {penalty:.1f}ms")
    print()

    return {"sticky_p50": sticky_p50, "evicted_p50": evicted_p50, "penalty": penalty}


# ---------------------------------------------------------------------------
# Experiment 4: Actual KVE from Worker Metrics
# ---------------------------------------------------------------------------

def experiment_actual_kve(worker_metrics_port: int, num_workers: int):
    """Scrape current worker metrics to compute actual KV cache efficiency."""
    print("=" * 60)
    print("Experiment 4: Actual KVE from Worker Prometheus Metrics")
    print("=" * 60)

    metrics = _scrape_worker_prefill(worker_metrics_port, num_workers)

    total_cache = 0.0
    total_compute = 0.0
    for wid, m in metrics.items():
        cache = m["cache"]
        compute = m["compute"]
        total = cache + compute
        kve = cache / total * 100 if total > 0 else 0
        print(f"  Worker {wid}: cache={cache:,.0f}  compute={compute:,.0f}  KVE={kve:.1f}%")
        total_cache += cache
        total_compute += compute

    overall_kve = total_cache / (total_cache + total_compute) * 100 if (total_cache + total_compute) > 0 else 0
    print(f"\n  Overall KVE: {overall_kve:.1f}%  (cache={total_cache:,.0f}  compute={total_compute:,.0f})")
    print()

    return {"overall_kve": overall_kve, "total_cache": total_cache, "total_compute": total_compute, "per_worker": metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Routing Efficiency Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8099")
    parser.add_argument("--model", default="Llama-3.3-70B-Instruct")
    parser.add_argument("--worker-metrics-port", type=int, default=18081)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    results = {}

    # Experiment 1
    results["cold_vs_warm"] = experiment_cold_vs_warm(
        args.base_url, args.model, args.reps,
        args.worker_metrics_port, args.num_workers)

    # Experiment 2
    results["agentic_turns"] = experiment_agentic_turns(args.base_url, args.model, args.turns)

    # Experiment 3
    results["cross_worker"] = experiment_cross_worker_penalty(args.base_url, args.model, args.reps)

    # Experiment 4
    results["actual_kve"] = experiment_actual_kve(args.worker_metrics_port, args.num_workers)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    cw = results["cold_vs_warm"]
    xw = results["cross_worker"]
    kve = results["actual_kve"]

    print(f"  Value of correct routing (cold→warm): {cw['delta']:.1f}ms TTFT savings")
    print(f"  Cache eviction penalty:               {xw['penalty']:.1f}ms TTFT cost")
    print(f"  Actual overall KVE:                   {kve['overall_kve']:.1f}%")

    # Estimate ideal KVE for agentic workload (30 calls/scenario, first cold)
    ideal_kve = 93.7  # from plan analysis
    gap = ideal_kve - kve["overall_kve"]
    if gap > 0:
        avg_prompt = 10000  # typical prompt tokens
        wasted_tokens = avg_prompt * gap / 100
        wasted_ms = wasted_tokens * 0.015  # ~0.015ms per token prefill
        print(f"  Ideal KVE (30 calls/scenario):        {ideal_kve:.1f}%")
        print(f"  KVE gap (room for improvement):       {gap:.1f}%")
        print(f"  Estimated wasted compute:             ~{wasted_tokens:.0f} tokens/request")
        print(f"  Estimated wasted TTFT:                ~{wasted_ms:.1f}ms/request")

    print()

    if args.output:
        # Remove non-serializable items
        clean = json.loads(json.dumps(results, default=str))
        Path(args.output).write_text(json.dumps(clean, indent=2))
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
