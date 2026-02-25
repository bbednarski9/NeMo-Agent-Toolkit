#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Bare-metal Dynamo with Cache Pinning (HiCache + PIN)
#
# Runs Dynamo frontend + SGLang workers directly on the host.
# Requires: etcd + NATS running in Docker (see deploy/docker-compose.yml).
# Requires: dynamo_pin venv with Dynamo + SGLang pin fork installed.
#
# Based on: https://gitlab-master.nvidia.com/idhanani/sgl-dyn-cache-pin/dynamo-stack.sh
# Adapted for: 4x B200, TP=2, 2 workers, Llama-3.3-70B-Instruct
#
# Usage:
#   bash start_dynamo_cache_pinning.sh
#   # Ctrl+C to stop all processes
#
# Logs: /tmp/dynamo-cache-pin/all.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env for configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# ── Config (from .env or defaults) ──────────────────────────────────────────
MODEL_DIR="${DYNAMO_MODEL_DIR:-$HOME/models/Llama-3.3-70B-Instruct}"
MODEL_NAME="$(basename "$MODEL_DIR")"
HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
TP_SIZE="${DYNAMO_TP_SIZE:-2}"
GPU_DEVICES="${DYNAMO_GPU_DEVICES:-0,1,2,3}"
PAGE_SIZE="${DYNAMO_KV_BLOCK_SIZE:-16}"
MEM_FRACTION="${DYNAMO_MEM_FRACTION_STATIC:-0.45}"
WORKER_METRICS_PORT="${DYNAMO_WORKER_METRICS_PORT:-18081}"

# HiCache config
HICACHE_RATIO="${DYNAMO_HICACHE_RATIO:-1.0}"
HICACHE_POLICY="${DYNAMO_HICACHE_POLICY:-write_through}"

# Infrastructure ports
ETCD_PORT="${DYNAMO_ETCD_PORT:-2379}"
NATS_PORT="${DYNAMO_NATS_PORT:-4222}"

# Venv
VENV="${DYNAMO_PIN_VENV:-$HOME/.venvs/dynamo_pin}"

LOG_DIR="/tmp/dynamo-cache-pin"
PIDFILE="$LOG_DIR/pids"

# ── Derived config ──────────────────────────────────────────────────────────
NUM_GPUS=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)
NUM_WORKERS=$((NUM_GPUS / TP_SIZE))
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"

# ── Validation ──────────────────────────────────────────────────────────────
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: Venv not found at $VENV" >&2
    echo "  Build it first: see the bare-metal cache pin testing plan" >&2
    exit 1
fi
if [ $((NUM_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "ERROR: NUM_GPUS ($NUM_GPUS) not divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi

echo "========================================================="
echo "Dynamo Cache Pinning (Bare-Metal)"
echo "========================================================="
echo "Model:       $MODEL_NAME ($MODEL_DIR)"
echo "GPUs:        $GPU_DEVICES ($NUM_GPUS total, TP=$TP_SIZE, $NUM_WORKERS workers)"
echo "HTTP Port:   $HTTP_PORT"
echo "Page Size:   $PAGE_SIZE"
echo "Mem Fraction: $MEM_FRACTION"
echo "HiCache:     ratio=$HICACHE_RATIO, policy=$HICACHE_POLICY"
echo "Venv:        $VENV"
echo "Log Dir:     $LOG_DIR"
echo "========================================================="

# ── Activate venv ───────────────────────────────────────────────────────────
source "$VENV/bin/activate"

# ── Cleanup ─────────────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -f "$PIDFILE"
    echo "Done. Logs in $LOG_DIR/"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
> "$PIDFILE"

# ── Preflight ───────────────────────────────────────────────────────────────
echo ""
echo "Checking infrastructure..."
curl -sf "http://localhost:$ETCD_PORT/health" >/dev/null 2>&1 || {
    echo "ERROR: etcd not running at localhost:$ETCD_PORT" >&2
    echo "  Start it: cd /raid/bbednarski/dynamo/deploy && docker compose up -d" >&2
    exit 1
}
echo "  etcd: OK (localhost:$ETCD_PORT)"

# NATS health check (try both monitoring port and client port)
if curl -sf "http://localhost:8222/healthz" >/dev/null 2>&1; then
    echo "  NATS: OK (localhost:$NATS_PORT, monitoring on 8222)"
elif timeout 2 bash -c "</dev/tcp/localhost/$NATS_PORT" 2>/dev/null; then
    echo "  NATS: OK (localhost:$NATS_PORT, TCP reachable)"
else
    echo "ERROR: NATS not running at localhost:$NATS_PORT" >&2
    echo "  Start it: cd /raid/bbednarski/dynamo/deploy && docker compose up -d" >&2
    exit 1
fi

LOGFILE="$LOG_DIR/all.log"
> "$LOGFILE"

# ── Frontend ────────────────────────────────────────────────────────────────
echo ""
echo "Starting frontend (port $HTTP_PORT, KV routing, HiCache on workers)..."

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_DIR" \
    --namespace workers \
    --router-mode kv \
    --kv-cache-block-size "$PAGE_SIZE" \
    --router-reset-states \
    2>&1 | tee -a "$LOGFILE" &
PIDS+=($!)
echo "${PIDS[-1]}" >> "$PIDFILE"
echo "  Frontend PID: ${PIDS[-1]}"

# ── Workers ─────────────────────────────────────────────────────────────────
echo ""
echo "Starting $NUM_WORKERS workers (TP=$TP_SIZE each)..."

for ((i=0; i<NUM_WORKERS; i++)); do
    START_GPU=$((i * TP_SIZE))
    WORKER_GPUS=""
    for ((g=START_GPU; g<START_GPU+TP_SIZE; g++)); do
        [ -n "$WORKER_GPUS" ] && WORKER_GPUS+=","
        WORKER_GPUS+="${GPU_ARRAY[$g]}"
    done

    WORKER_PORT=$((30000 + i))
    METRICS_PORT=$((WORKER_METRICS_PORT + i))
    ZMQ_PORT=$((20080 + i))

    echo "  Worker $i: GPUs=$WORKER_GPUS, port=$WORKER_PORT, metrics=$METRICS_PORT, zmq=$ZMQ_PORT"

    CUDA_VISIBLE_DEVICES="$WORKER_GPUS" \
    OTEL_SERVICE_NAME="dynamo-worker-$i" \
    DYN_SYSTEM_PORT="$METRICS_PORT" \
    DYN_NAMESPACE=workers \
    python3 -m dynamo.sglang \
        --model-path "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --page-size "$PAGE_SIZE" \
        --tp "$TP_SIZE" \
        --mem-fraction-static "$MEM_FRACTION" \
        --trust-remote-code \
        --enable-metrics \
        --enable-hierarchical-cache \
        --hicache-write-policy "$HICACHE_POLICY" \
        --hicache-ratio "$HICACHE_RATIO" \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$ZMQ_PORT"'"}' \
        2>&1 | tee -a "$LOGFILE" &
    PIDS+=($!)
    echo "${PIDS[-1]}" >> "$PIDFILE"
    echo "  Worker $i PID: ${PIDS[-1]}"
done

echo ""
echo "========================================================="
echo "All processes started. Waiting for model to load..."
echo "========================================================="
echo ""
echo "Monitor logs:     tail -f $LOGFILE"
echo "API endpoint:     http://localhost:$HTTP_PORT/v1/chat/completions"
echo "Health check:     curl http://localhost:$HTTP_PORT/health"
echo "Worker metrics:   http://localhost:$WORKER_METRICS_PORT/metrics"
echo ""
echo "Test with cache_control:"
echo "  curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":10,\"stream\":true,\"nvext\":{\"cache_control\":{\"type\":\"ephemeral\",\"ttl\":\"5m\"}}}'"
echo ""
echo "Ctrl+C to stop all processes."
echo "========================================================="

# Wait for any process to exit (on Ctrl+C, cleanup() fires)
wait
