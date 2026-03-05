#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Router Pretraining Pipeline
#
# Orchestrates the full pretraining flow in the correct order:
#
#   Step 1: Health check
#   Step 2: Build prediction tries — nat eval on *_train.yml configs (deterministic,
#            no router state dependency). Each domain produces a per-call statistics
#            trie used to give the router accurate OSL/IAT/remaining-call hints.
#   Step 3: Collect prediction tries into data/prediction_tries/
#   Step 4: Optimize router params — nat optimize on optimize_all_v0.yml, now
#            using the tries for realistic routing hints per trial.
#   Step 5: Final training run — reset + nat eval on *_train.yml with optimized
#            params to accumulate the full learner state (Beta + LinTS).
#   Step 6: Save learner state + router config
#
# Prerequisites:
#   - Dynamo stack running with learner management HTTP server enabled
#   - Training datasets present (agent_leaderboard_v2_*_split0_pct20of100.json)
#   - NAT eval venv activated
#
# Usage:
#   bash run_pretrain_v0.sh [--config-dir <path>] [--mgmt-port <port>]
#                           [--skip-trie] [--skip-optimize] [--skip-final-train]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG_DIR="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train"
OUTPUT_BASE="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization"
PREDICTION_TRIES_DIR="$REPO_ROOT/examples/dynamo_integration/data/prediction_tries"
TRAIN_OUTPUT_BASE="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain_test_train"

MGMT_PORT=8084
SKIP_TRIE=false
SKIP_OPTIMIZE=false
SKIP_FINAL_TRAIN=false

DOMAINS="banking healthcare insurance investment telecom"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --mgmt-port)
            MGMT_PORT="$2"
            shift 2
            ;;
        --skip-trie)
            SKIP_TRIE=true
            shift
            ;;
        --skip-optimize)
            SKIP_OPTIMIZE=true
            shift
            ;;
        --skip-final-train)
            SKIP_FINAL_TRAIN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config-dir <path>] [--mgmt-port <port>]"
            echo "          [--skip-trie] [--skip-optimize] [--skip-final-train]"
            exit 1
            ;;
    esac
done

MGMT_URL="http://localhost:${MGMT_PORT}"

echo "========================================================="
echo "Router Pretraining Pipeline"
echo "========================================================="
echo "Management URL:  $MGMT_URL"
echo "Config dir:      $CONFIG_DIR"
echo "Output base:     $OUTPUT_BASE"
echo ""

# ── Step 1: Health check ─────────────────────────────────────────────────
echo "Step 1: Verifying Dynamo stack health..."
if ! curl -sf "$MGMT_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Learner management server not responding at $MGMT_URL/health"
    echo "Make sure the Dynamo stack is running with LEARNER_STATE_PORT=$MGMT_PORT"
    exit 1
fi
echo "  Learner management server is healthy"
echo ""

# ── Step 2: Build prediction tries (deterministic) ───────────────────────
if [[ "$SKIP_TRIE" == "false" ]]; then
    echo "Step 2: Building prediction tries from training data..."
    echo "  Running *_train.yml configs concurrently (builds per-domain tries)..."

    # Reset learner state so trie-building runs start clean
    curl -sf -X POST "$MGMT_URL/state/reset" > /dev/null

    PIDS=()
    cd "$REPO_ROOT"
    for domain in $DOMAINS; do
        train_cfg="$CONFIG_DIR/${domain}_train.yml"
        if [[ ! -f "$train_cfg" ]]; then
            echo "  SKIP $domain: config not found: $train_cfg"
            continue
        fi
        echo "  Starting $domain..."
        nat eval --config_file "$train_cfg" > "/tmp/pretrain_trie_${domain}.log" 2>&1 &
        PIDS+=($!)
    done

    echo "  Waiting for all trie-building evals to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  WARNING: one eval failed (check /tmp/pretrain_trie_*.log)"
    done
    echo "  Trie building complete"
    echo ""
else
    echo "Step 2: SKIPPED (--skip-trie)"
    echo ""
fi

# ── Step 3: Collect prediction tries ─────────────────────────────────────
echo "Step 3: Collecting prediction tries from $TRAIN_OUTPUT_BASE..."
bash "$SCRIPT_DIR/collect_prediction_tries.sh" --output-base "$TRAIN_OUTPUT_BASE"
echo ""

# ── Step 4: Optimize router params (using tries for realistic hints) ──────
if [[ "$SKIP_OPTIMIZE" == "false" ]]; then
    OPT_CONFIG="$CONFIG_DIR/optimize_all_v0.yml"

    if [[ ! -f "$OPT_CONFIG" ]]; then
        echo "ERROR: Optimize config not found: $OPT_CONFIG"
        exit 1
    fi

    echo "Step 4: Running router parameter optimization (with trie hints)..."
    echo "  Config: $OPT_CONFIG"

    # Optimizer resets state before each trial — no manual reset needed here
    cd "$REPO_ROOT"
    nat optimize --config_file "$OPT_CONFIG" 2>&1 \
        | tee "/tmp/pretrain_optimize_all.log" || {
        echo "  WARNING: optimization failed (exit code: $?)"
        echo "  Check /tmp/pretrain_optimize_all.log"
    }
    echo "  Optimization complete"
    echo ""
else
    echo "Step 4: SKIPPED (--skip-optimize)"
    echo ""
fi

# ── Step 5: Final training run (accumulate learner state) ─────────────────
if [[ "$SKIP_FINAL_TRAIN" == "false" ]]; then
    echo "Step 5: Final training run (building accumulated learner state)..."

    # Reset to pristine before accumulating learner state with optimized params
    echo "  Resetting learner state..."
    curl -sf -X POST "$MGMT_URL/state/reset" > /dev/null

    PIDS=()
    cd "$REPO_ROOT"
    for domain in $DOMAINS; do
        train_cfg="$CONFIG_DIR/${domain}_train.yml"
        echo "  Running eval: $domain..."
        nat eval --config_file "$train_cfg" > "/tmp/pretrain_final_${domain}.log" 2>&1 &
        PIDS+=($!)
    done

    echo "  Waiting for all domain evals to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  WARNING: one eval failed (check /tmp/pretrain_final_*.log)"
    done
    echo "  Final training complete"
    echo ""
else
    echo "Step 5: SKIPPED (--skip-final-train)"
    echo ""
fi

# ── Step 6: Save artifacts ────────────────────────────────────────────────
echo "Step 6: Saving artifacts..."
mkdir -p "$OUTPUT_BASE"

LEARNER_STATE_FILE="$OUTPUT_BASE/learner_state.json"
curl -sf "$MGMT_URL/state" -o "$LEARNER_STATE_FILE"
echo "  Learner state -> $LEARNER_STATE_FILE"

ROUTER_CONFIG_FILE="$OUTPUT_BASE/router_config.json"
curl -sf "$MGMT_URL/config" -o "$ROUTER_CONFIG_FILE"
echo "  Router config  -> $ROUTER_CONFIG_FILE"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "========================================================="
echo "Pretraining Pipeline Complete"
echo "========================================================="
echo ""
echo "Artifacts:"
echo "  Learner state:    $LEARNER_STATE_FILE"
echo "  Router config:    $ROUTER_CONFIG_FILE"
echo "  Prediction tries: $PREDICTION_TRIES_DIR/"
ls "$PREDICTION_TRIES_DIR"/*.json 2>/dev/null | while read -r f; do
    echo "    $(basename "$f")"
done
echo ""
echo "To run held-out evaluation:"
echo "  curl -sf -X POST $MGMT_URL/state/reset"
echo "  curl -sf -X POST $MGMT_URL/state  -H 'Content-Type: application/json' -d @$LEARNER_STATE_FILE"
echo "  curl -sf -X POST $MGMT_URL/config -H 'Content-Type: application/json' -d @$ROUTER_CONFIG_FILE"
echo "  bash $SCRIPT_DIR/run_multi_domain_benchmark.sh \\"
echo "    --configs $CONFIG_DIR/*_eval.yml"
