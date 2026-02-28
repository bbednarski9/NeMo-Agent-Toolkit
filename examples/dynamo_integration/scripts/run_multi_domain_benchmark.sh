#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-Domain Router Comparison Benchmark
#
# Launches concurrent eval processes for every config in the given config dir
# against the same Dynamo endpoint. Each domain has a unique system prompt
# (~7500 tokens of domain-specific tool definitions), creating realistic KV
# cache pressure.
#
# At the start of each run:
#   - Domain order is shuffled
#   - Config variants within each domain (v0-v4) are shuffled
# This ensures varied prompt interleaving across benchmark runs.
#
# Prerequisites:
#   - Dynamo stack running (start_dynamo_optimized_thompson_hints_sglang.sh)
#   - All 5 domains downloaded (download_agent_leaderboard_v2.py --domains all)
#   - NAT eval venv activated
#
# Usage:
#   bash run_multi_domain_benchmark.sh [--config-dir <path>] [--concurrency <n>]
#
# Default concurrency: 4 per config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults
CONCURRENCY=4
CONFIG_DIR="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/configs/multi_domain"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config-dir <path>] [--concurrency <n>]"
            exit 1
            ;;
    esac
done

# Resolve config dir relative to repo root if not absolute
if [[ "$CONFIG_DIR" != /* ]]; then
    CONFIG_DIR="$REPO_ROOT/$CONFIG_DIR"
fi

DOMAINS="banking healthcare insurance investment telecom"

# ── Shuffle domain order, then shuffle variants within each domain ────────────
# Produces a flat list of config paths: all variants of domain[0] (shuffled),
# then all variants of domain[1] (shuffled), etc., with domains themselves shuffled.
SHUFFLED_CONFIGS=$(python3 -c "
import sys, random, glob, os

domains = '$DOMAINS'.split()
config_dir = '$CONFIG_DIR'

random.shuffle(domains)
result = []
for domain in domains:
    variants = sorted(glob.glob(os.path.join(config_dir, f'{domain}_v*.yml')))
    if not variants:
        print(f'ERROR: No configs for domain {domain!r} in {config_dir}', file=sys.stderr)
        sys.exit(1)
    random.shuffle(variants)
    result.extend(variants)

print('\n'.join(result))
")

# Convert to array
mapfile -t ALL_CONFIGS <<< "$SHUFFLED_CONFIGS"
N_CONFIGS=${#ALL_CONFIGS[@]}

echo "========================================================="
echo "Multi-Domain Router Comparison Benchmark"
echo "========================================================="
echo "Config dir:               $CONFIG_DIR"
echo "Concurrency per config:   $CONCURRENCY"
echo "Total configs:            $N_CONFIGS"
echo "Total concurrent sessions: $((CONCURRENCY * N_CONFIGS))"
echo ""
echo "Launch order (shuffled domains, shuffled variants within domain):"
for cfg in "${ALL_CONFIGS[@]}"; do
    echo "  $(basename "$cfg" .yml)"
done
echo ""

# Verify service is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Dynamo service not responding at localhost:8000"
    exit 1
fi
echo "✓ Dynamo service is healthy"

# Scrape baseline metrics
echo "Scraping baseline worker metrics..."
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

# Override concurrency in all configs if not default
if [ "$CONCURRENCY" != "4" ]; then
    echo "Setting concurrency to $CONCURRENCY in all configs..."
    for cfg in "${ALL_CONFIGS[@]}"; do
        sed -i "s/max_concurrency: [0-9]*/max_concurrency: $CONCURRENCY/" "$cfg"
    done
fi

# Launch all configs concurrently (from repo root so relative paths resolve)
cd "$REPO_ROOT"
PIDS=()
echo "Launching eval processes..."
for cfg in "${ALL_CONFIGS[@]}"; do
    variant=$(basename "$cfg" .yml)
    echo "  Starting $variant..."
    nat eval --config_file "$cfg" > "/tmp/multi_domain_${variant}.log" 2>&1 &
    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "All configs launched. Waiting for completion..."
echo "  Monitor with: tail -f /tmp/multi_domain_*.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!ALL_CONFIGS[@]}"; do
    variant=$(basename "${ALL_CONFIGS[$i]}" .yml)
    if wait "${PIDS[$i]}"; then
        echo "  ✓ $variant completed successfully"
    else
        echo "  ✗ $variant failed (exit code: $?)"
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
    echo "⚠ $FAILED config(s) failed. Check logs in /tmp/multi_domain_*.log"
else
    echo "✓ All configs completed successfully"
fi

# Collect output dirs for plotting
OUTPUT_DIRS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    out_dir=$(python3 -c "
import yaml, sys
with open('$cfg') as f:
    c = yaml.safe_load(f)
print(c['eval']['general']['output']['dir'])
" 2>/dev/null || echo "")
    if [[ -n "$out_dir" ]]; then
        OUTPUT_DIRS+=("$REPO_ROOT/$out_dir")
    fi
done

echo ""
echo "Results in:"
for cfg in "${ALL_CONFIGS[@]}"; do
    variant=$(basename "$cfg" .yml)
    echo "  $CONFIG_DIR/../outputs/dynamo_evals/multi_domain/${variant}/"
done

COMPARISON_DIR="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain/comparison"

if [ ${#OUTPUT_DIRS[@]} -gt 0 ]; then
    echo ""
    echo "Generating comparison plots..."
    python3 "$SCRIPT_DIR/plot_throughput_vs_tsq_per_request.py" \
        "${OUTPUT_DIRS[@]}" \
        --output "$COMPARISON_DIR" \
        --consolidate-legend

    python3 "$SCRIPT_DIR/plot_throughput_histograms_per_request.py" \
        "${OUTPUT_DIRS[@]}" \
        --output "$COMPARISON_DIR" \
        --consolidate-legend

    echo "  Plots saved to: $COMPARISON_DIR"
else
    echo ""
    echo "Generate comparison plots with:"
    echo "  python scripts/plot_throughput_vs_tsq_per_request.py \\"
    for cfg in "${ALL_CONFIGS[@]}"; do
        variant=$(basename "$cfg" .yml)
        echo "    ./react_benchmark_agent/outputs/dynamo_evals/multi_domain/${variant} \\"
    done
    echo "    --output ./react_benchmark_agent/outputs/dynamo_evals/multi_domain/comparison \\"
    echo "    --consolidate-legend"
fi
