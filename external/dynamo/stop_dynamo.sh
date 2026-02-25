#!/bin/bash
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

# Dynamo Shutdown Script
# Stops all components: Docker containers AND bare-metal processes
# Works for: UNIFIED, THOMPSON SAMPLING, DISAGGREGATED, and CACHE PINNING modes
# Supports both SGLang and vLLM backends
#
# Usage:
#   bash stop_dynamo.sh                  # Stop Dynamo, ETCD, NATS + bare-metal processes
#   bash stop_dynamo.sh --kill-metrics   # Also stop Prometheus and Grafana
#   bash stop_dynamo.sh --clear-metrics  # Stop monitoring stack AND remove Prometheus data volume

# Parse command line arguments
KILL_METRICS=false
CLEAR_METRICS=false
for arg in "$@"; do
    case $arg in
        --kill-metrics)
            KILL_METRICS=true
            shift
            ;;
        --clear-metrics)
            KILL_METRICS=true
            CLEAR_METRICS=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash stop_dynamo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --kill-metrics     Also stop Prometheus and Grafana containers"
            echo "  --clear-metrics    Stop monitoring stack AND remove Prometheus data volume (clears old metrics)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
    esac
done

echo "========================================================="
echo "Stopping Dynamo FULL STACK (SGLang/vLLM)"
echo "========================================================="
echo ""

# Stop Dynamo containers (check for SGLang and vLLM variants)
STOPPED_CONTAINER=false

# SGLang containers
if docker ps --format '{{.Names}}' | grep -q "^dynamo-sglang$"; then
    echo "Stopping Dynamo container (SGLang)..."
    docker stop dynamo-sglang
    docker rm dynamo-sglang
    echo "✓ Dynamo SGLang container stopped and removed"
    STOPPED_CONTAINER=true
fi

if docker ps --format '{{.Names}}' | grep -q "^dynamo-sglang-thompson$"; then
    echo "Stopping Dynamo container (SGLang Thompson Sampling)..."
    docker stop dynamo-sglang-thompson
    docker rm dynamo-sglang-thompson
    echo "✓ Dynamo SGLang Thompson container stopped and removed"
    STOPPED_CONTAINER=true
fi

# vLLM containers
if docker ps --format '{{.Names}}' | grep -q "^dynamo-vllm$"; then
    echo "Stopping Dynamo container (vLLM)..."
    docker stop dynamo-vllm
    docker rm dynamo-vllm
    echo "✓ Dynamo vLLM container stopped and removed"
    STOPPED_CONTAINER=true
fi

if [ "$STOPPED_CONTAINER" = false ]; then
    echo "  (No Dynamo containers running)"
fi

# Stop ETCD
if docker ps --format '{{.Names}}' | grep -q "^etcd-dynamo$"; then
    echo ""
    echo "Stopping ETCD container..."
    docker stop etcd-dynamo
    docker rm etcd-dynamo
    echo "✓ ETCD container stopped and removed"
else
    echo "  (ETCD container not running)"
fi

# Stop NATS
if docker ps --format '{{.Names}}' | grep -q "^nats-dynamo$"; then
    echo ""
    echo "Stopping NATS container..."
    docker stop nats-dynamo
    docker rm nats-dynamo
    echo "✓ NATS container stopped and removed"
else
    echo "  (NATS container not running)"
fi

# ── Bare-metal process cleanup ───────────────────────────────────────────────
# Kills dynamo processes launched directly on the host (not in containers).
# Uses PID file from start_dynamo_cache_pinning.sh, then falls back to pgrep.
BAREMETAL_PIDFILE="/tmp/dynamo-cache-pin/pids"
KILLED_BAREMETAL=false

kill_process_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child"
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
}

echo ""
echo "========================================================="
echo "Checking for bare-metal Dynamo processes..."
echo "========================================================="

# Phase 1: Kill from PID file (most targeted)
if [ -f "$BAREMETAL_PIDFILE" ]; then
    echo "  Found PID file: $BAREMETAL_PIDFILE"
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing process tree rooted at PID $pid..."
            kill_process_tree "$pid"
            KILLED_BAREMETAL=true
        fi
    done < "$BAREMETAL_PIDFILE"
    rm -f "$BAREMETAL_PIDFILE"
fi

# Phase 2: pgrep fallback for any orphaned dynamo processes
DYNAMO_PATTERNS=(
    "dynamo\.frontend"
    "dynamo\.sglang"
    "dynamo\.vllm"
)

for pattern in "${DYNAMO_PATTERNS[@]}"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    for pid in $pids; do
        CMDLINE=$(ps -p "$pid" -o args= 2>/dev/null || true)
        echo "  Found orphaned process: PID $pid ($CMDLINE)"
        kill_process_tree "$pid"
        KILLED_BAREMETAL=true
    done
done

if [ "$KILLED_BAREMETAL" = true ]; then
    sleep 2
    # SIGKILL any survivors
    SURVIVORS=false
    for pattern in "${DYNAMO_PATTERNS[@]}"; do
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        for pid in $pids; do
            echo "  Force-killing stubborn process: PID $pid"
            kill -9 "$pid" 2>/dev/null || true
            SURVIVORS=true
        done
    done
    if [ -f "$BAREMETAL_PIDFILE" ]; then
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force-killing stubborn process: PID $pid"
                kill -9 "$pid" 2>/dev/null || true
                SURVIVORS=true
            fi
        done < "$BAREMETAL_PIDFILE"
    fi
    echo "  Done killing bare-metal processes."
else
    echo "  (No bare-metal Dynamo processes found)"
fi

# Stop monitoring stack if --kill-metrics flag is set
if [ "$KILL_METRICS" = true ]; then
    echo ""
    echo "========================================================="
    echo "Stopping Monitoring Stack (--kill-metrics)"
    echo "========================================================="
    
    # Stop Prometheus
    if docker ps --format '{{.Names}}' | grep -q "^dynamo-prometheus$"; then
        echo ""
        echo "Stopping Prometheus container..."
        docker stop dynamo-prometheus
        docker rm dynamo-prometheus
        echo "✓ Prometheus container stopped and removed"
    else
        echo "  (Prometheus container not running)"
    fi
    
    # Stop Grafana
    if docker ps --format '{{.Names}}' | grep -q "^dynamo-grafana$"; then
        echo ""
        echo "Stopping Grafana container..."
        docker stop dynamo-grafana
        docker rm dynamo-grafana
        echo "✓ Grafana container stopped and removed"
    else
        echo "  (Grafana container not running)"
    fi
    
    # Clear Prometheus data volume if --clear-metrics flag is set
    if [ "$CLEAR_METRICS" = true ]; then
        echo ""
        echo "Clearing Prometheus data volume..."
        docker volume rm monitoring_prometheus_data && echo "✓ Prometheus data volume removed (old metrics cleared)"
    fi
fi

echo ""
echo "========================================================="
echo "✓ All components stopped!"
if [ "$KILL_METRICS" = true ]; then
    echo "  (including monitoring stack)"
fi
if [ "$CLEAR_METRICS" = true ]; then
    echo "  (Prometheus data volume cleared)"
fi
echo "========================================================="
echo ""
echo "To restart:"
echo "  Standard Unified:     bash start_dynamo_unified.sh"
echo "  SGLang Thompson:      bash start_dynamo_optimized_thompson_hints_sglang.sh"
echo "  vLLM Thompson:        bash start_dynamo_optimized_thompson_hints_vllm.sh"
echo "  Cache Pinning:        bash start_dynamo_cache_pinning.sh"
echo ""

