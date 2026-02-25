#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-Domain Router Comparison Benchmark
#
# Launches 5 concurrent eval processes (one per domain) against the same
# Dynamo endpoint. Each domain has a unique system prompt (~7500 tokens of
# domain-specific tool definitions), creating realistic KV cache pressure.
#
# Prerequisites:
#   - Dynamo stack running (start_dynamo_optimized_thompson_hints_sglang.sh)
#   - All 5 domains downloaded (download_agent_leaderboard_v2.py --domains all)
#   - NAT eval venv activated
#
# Usage:
#   bash run_multi_domain_benchmark.sh [concurrency_per_domain]
#
# Default concurrency: 4 per domain (20 total concurrent sessions across 5 domains)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/configs/multi_domain"

CONCURRENCY="${1:-4}"
DOMAINS="banking healthcare insurance investment telecom"

echo "========================================================="
echo "Multi-Domain Router Comparison Benchmark"
echo "========================================================="
echo "Domains: $DOMAINS"
echo "Concurrency per domain: $CONCURRENCY"
echo "Total concurrent sessions: $((CONCURRENCY * 5))"
echo ""

# Verify service is running
if ! curl -s http://localhost:8099/health > /dev/null 2>&1; then
    echo "ERROR: Dynamo service not responding at localhost:8099"
    exit 1
fi
echo "✓ Dynamo service is healthy"

# Scrape baseline metrics
echo "Scraping baseline worker metrics..."
python3 -c "
from urllib.request import urlopen
import json
for i in range(8):
    port = 18081 + i
    try:
        body = urlopen(f'http://localhost:{port}/metrics', timeout=2).read().decode()
        cache = compute = 0
        for line in body.splitlines():
            if 'prefill_cache' in line and 'realtime_tokens' in line and not line.startswith('#'):
                cache = float(line.rsplit(' ', 1)[-1])
            elif 'prefill_compute' in line and 'realtime_tokens' in line and not line.startswith('#'):
                compute = float(line.rsplit(' ', 1)[-1])
        total = cache + compute
        kve = cache / total * 100 if total > 0 else 0
        print(f'  Worker {i}: cache={cache:,.0f}  compute={compute:,.0f}  KVE={kve:.1f}%')
    except Exception:
        break
"
echo ""

# Override concurrency in configs if specified
if [ "$CONCURRENCY" != "4" ]; then
    echo "Setting concurrency to $CONCURRENCY in all configs..."
    for domain in $DOMAINS; do
        sed -i "s/max_concurrency: [0-9]*/max_concurrency: $CONCURRENCY/" "$CONFIG_DIR/$domain.yml"
    done
fi

# Launch all domains concurrently (from repo root so relative paths resolve)
cd "$REPO_ROOT"
PIDS=()
echo "Launching eval processes..."
for domain in $DOMAINS; do
    echo "  Starting $domain..."
    nat eval --config_file "$CONFIG_DIR/$domain.yml" > "/tmp/multi_domain_${domain}.log" 2>&1 &
    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "All domains launched. Waiting for completion..."
echo "  Monitor with: tail -f /tmp/multi_domain_*.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    domain=$(echo "$DOMAINS" | cut -d' ' -f$((i+1)))
    if wait "${PIDS[$i]}"; then
        echo "  ✓ $domain completed successfully"
    else
        echo "  ✗ $domain failed (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

# Scrape final metrics
echo "Scraping final worker metrics..."
python3 -c "
from urllib.request import urlopen
for i in range(8):
    port = 18081 + i
    try:
        body = urlopen(f'http://localhost:{port}/metrics', timeout=2).read().decode()
        cache = compute = 0
        for line in body.splitlines():
            if 'prefill_cache' in line and 'realtime_tokens' in line and not line.startswith('#'):
                cache = float(line.rsplit(' ', 1)[-1])
            elif 'prefill_compute' in line and 'realtime_tokens' in line and not line.startswith('#'):
                compute = float(line.rsplit(' ', 1)[-1])
        total = cache + compute
        kve = cache / total * 100 if total > 0 else 0
        print(f'  Worker {i}: cache={cache:,.0f}  compute={compute:,.0f}  KVE={kve:.1f}%')
    except Exception:
        break
"

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "⚠ $FAILED domain(s) failed. Check logs in /tmp/multi_domain_*.log"
else
    echo "✓ All domains completed successfully"
fi

echo ""
echo "Results in:"
for domain in $DOMAINS; do
    echo "  $CONFIG_DIR/../outputs/dynamo_evals/multi_domain/$domain/"
done

echo ""
echo "Generate comparison plots with:"
echo "  python scripts/plot_throughput_vs_tsq_per_request.py \\"
for domain in $DOMAINS; do
    echo "    ./react_benchmark_agent/outputs/dynamo_evals/multi_domain/$domain \\"
done
echo "    --output ./react_benchmark_agent/outputs/dynamo_evals/multi_domain/comparison"
