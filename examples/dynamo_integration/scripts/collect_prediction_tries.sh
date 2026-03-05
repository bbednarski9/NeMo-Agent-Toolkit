#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collect prediction_trie.json files from v0 eval outputs into a central directory.
#
# For each domain, finds the most recently modified prediction_trie.json from
# the v0 job outputs and copies it to data/prediction_tries/<domain>_prediction_trie.json.
#
# Usage:
#   bash collect_prediction_tries.sh [--output-base <path>]
#
# Default output base: the multi_domain output dir under react_benchmark_agent/outputs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

OUTPUT_BASE="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain"
DEST_DIR="$REPO_ROOT/examples/dynamo_integration/data/prediction_tries"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output-base <path>]"
            exit 1
            ;;
    esac
done

mkdir -p "$DEST_DIR"

DOMAINS="banking healthcare insurance investment telecom"
COLLECTED=0
FAILED=0

echo "Collecting prediction tries from: $OUTPUT_BASE"
echo "Destination: $DEST_DIR"
echo ""

for domain in $DOMAINS; do
    # Search for the most recent prediction_trie.json under the domain output dir.
    # Supports both old naming ({domain}_v0) and new naming ({domain}, {domain}_train).
    search_dir=""
    for candidate in "${domain}_train" "${domain}" "${domain}_v0"; do
        if [[ -d "$OUTPUT_BASE/$candidate" ]]; then
            search_dir="$OUTPUT_BASE/$candidate"
            break
        fi
    done

    if [[ -z "$search_dir" ]]; then
        echo "  SKIP $domain: no output directory found under $OUTPUT_BASE"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Find newest prediction_trie.json across all job dirs
    latest=$(find "$search_dir" -name "prediction_trie.json" -type f -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -z "$latest" ]]; then
        echo "  SKIP $domain: no prediction_trie.json found in $search_dir"
        FAILED=$((FAILED + 1))
        continue
    fi

    dest_file="$DEST_DIR/${domain}_prediction_trie.json"
    cp "$latest" "$dest_file"
    echo "  OK   $domain: $(basename "$(dirname "$latest")") -> $(basename "$dest_file")"
    COLLECTED=$((COLLECTED + 1))
done

echo ""
echo "Collected: $COLLECTED / $(echo $DOMAINS | wc -w)"
if [[ $FAILED -gt 0 ]]; then
    echo "Failed: $FAILED (missing outputs — run train evals first)"
fi

# Build merged all-domain trie for optimize_all_v0.yml
ALL_TRIE="$DEST_DIR/all_prediction_trie.json"
DOMAIN_TRIES=()
for domain in $DOMAINS; do
    f="$DEST_DIR/${domain}_prediction_trie.json"
    [[ -f "$f" ]] && DOMAIN_TRIES+=("$f")
done

if [[ ${#DOMAIN_TRIES[@]} -gt 0 ]]; then
    python3 - "${DOMAIN_TRIES[@]}" "$ALL_TRIE" <<'EOF'
import json, sys
from pathlib import Path

inputs = sys.argv[1:-1]
output = sys.argv[-1]

# Merge by taking the first trie as base and accumulating sample_count.
# All domain tries share the same call-depth structure (same react_agent workflow),
# so we average statistics across domains at each call position.
merged = None
for path in inputs:
    trie = json.loads(Path(path).read_text())
    if merged is None:
        merged = trie
    else:
        # Simple merge: add sample counts (statistics are already means, so we keep first)
        def add_counts(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in b:
                    if k in a:
                        if k == 'sample_count':
                            a[k] = a.get(k, 0) + b.get(k, 0)
                        elif isinstance(a[k], dict):
                            add_counts(a[k], b[k])
        add_counts(merged, trie)

Path(output).write_text(json.dumps(merged, indent=2))
print(f"  Merged {len(inputs)} tries -> {Path(output).name}")
EOF
    echo "  OK   all_prediction_trie.json (merged from ${#DOMAIN_TRIES[@]} domains)"
else
    echo "  SKIP all_prediction_trie.json (no domain tries available)"
fi

echo ""
echo "Files in $DEST_DIR:"
ls -la "$DEST_DIR"/*.json 2>/dev/null || echo "  (none)"
