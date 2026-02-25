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

# Test Dynamo-native NvExt features in isolation
# Each test sends a request with specific nvext fields and verifies the response.
#
# Prerequisites:
#   - Dynamo is running (bash start_dynamo_unified.sh)
#   - For cache_control tests: DYNAMO_ENABLE_CACHE_CONTROL=true, DYNAMO_ENABLE_HIERARCHICAL_CACHE=true
#   - For osl tests: DYNAMO_ROUTER_TRACK_OUTPUT_BLOCKS=true
#   - For latency_sensitivity tests: DYNAMO_ROUTER_QUEUE_THRESHOLD=0.8
#
# Usage:
#   bash test_dynamo_native_features.sh [test_name]
#   bash test_dynamo_native_features.sh           # run all tests
#   bash test_dynamo_native_features.sh osl        # run only osl test
#   bash test_dynamo_native_features.sh priority   # run only priority test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

API_BASE="http://localhost:${DYNAMO_HTTP_PORT:-8099}"
MODEL_DIR="${DYNAMO_MODEL_DIR:-$HOME/models/Llama-3.3-70B-Instruct}"
MODEL_NAME="$(basename "$MODEL_DIR")"
CONTAINER_NAME="dynamo-sglang"

PASSED=0
FAILED=0
SKIPPED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }
skip() { echo "  SKIP: $1"; SKIPPED=$((SKIPPED + 1)); }

check_api() {
    if ! curl -sf "$API_BASE/health" > /dev/null 2>&1; then
        echo "ERROR: Dynamo API not reachable at $API_BASE/health"
        echo "  Start Dynamo first: bash start_dynamo_unified.sh"
        exit 1
    fi
    echo "API healthy at $API_BASE"
}

send_request() {
    local body="$1"
    local timeout="${2:-30}"
    curl -sf --max-time "$timeout" \
        -H "Content-Type: application/json" \
        "$API_BASE/v1/chat/completions" \
        -d "$body" 2>/dev/null
}

send_streaming_request() {
    local body="$1"
    local timeout="${2:-30}"
    curl -sf --max-time "$timeout" -N \
        -H "Content-Type: application/json" \
        "$API_BASE/v1/chat/completions" \
        -d "$body" 2>/dev/null
}

# ============================================================================
# Test 0: Baseline (no nvext)
# ============================================================================
test_baseline() {
    echo ""
    echo "=== Test 0: Baseline (no nvext) ==="
    local resp
    resp=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }')
    if [ $? -ne 0 ] || [ -z "$resp" ]; then
        fail "Baseline request failed"
        return
    fi
    local content
    content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content" ]; then
        pass "Baseline response received: $(echo "$content" | head -c 50)"
    else
        fail "Baseline response had no content"
        echo "  Response: $(echo "$resp" | head -c 200)"
    fi
}

# ============================================================================
# Test 1: osl (expected_output_tokens for router block tracking)
# ============================================================================
test_osl() {
    echo ""
    echo "=== Test 1: osl (Router Block Tracking) ==="
    echo "  Sending request with nvext.agent_hints.osl=512"

    local resp
    resp=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Write a short poem about caching."}],
        "max_tokens": 100,
        "nvext": {"agent_hints": {"osl": 512}}
    }')
    if [ $? -ne 0 ] || [ -z "$resp" ]; then
        fail "osl request failed"
        return
    fi
    local content
    content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content" ]; then
        pass "osl request accepted and returned content"
    else
        fail "osl request returned no content"
        echo "  Response: $(echo "$resp" | head -c 200)"
        return
    fi

    echo "  Checking container logs for expected_output_tokens..."
    local log_match
    log_match=$(timeout 5 docker logs --tail 200 "$CONTAINER_NAME" 2>&1 | grep -i "expected_output_tokens\|output.block\|osl" | tail -3 || true)
    if [ -n "$log_match" ]; then
        pass "Found osl-related log entries"
        echo "  $log_match"
    else
        skip "No osl log entries found (may need DYNAMO_ROUTER_TRACK_OUTPUT_BLOCKS=true or RUST_LOG=debug)"
    fi
}

# ============================================================================
# Test 2: latency_sensitivity (queue priority_jump)
# ============================================================================
test_latency_sensitivity() {
    echo ""
    echo "=== Test 2: latency_sensitivity (Queue Priority Jump) ==="
    echo "  Sending two concurrent requests with different latency_sensitivity values"

    local tmpdir
    tmpdir=$(mktemp -d)

    send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
        "max_tokens": 80,
        "nvext": {"agent_hints": {"latency_sensitivity": 0.1}}
    }' 60 > "$tmpdir/low_priority.json" 2>/dev/null &
    local pid_low=$!

    send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say the word hello."}],
        "max_tokens": 10,
        "nvext": {"agent_hints": {"latency_sensitivity": 100.0}}
    }' 60 > "$tmpdir/high_priority.json" 2>/dev/null &
    local pid_high=$!

    wait $pid_low 2>/dev/null
    wait $pid_high 2>/dev/null

    local low_ok high_ok
    low_ok=$(jq -r '.choices[0].message.content // empty' "$tmpdir/low_priority.json" 2>/dev/null)
    high_ok=$(jq -r '.choices[0].message.content // empty' "$tmpdir/high_priority.json" 2>/dev/null)

    if [ -n "$low_ok" ] && [ -n "$high_ok" ]; then
        pass "Both latency_sensitivity requests completed successfully"
    else
        fail "One or both latency_sensitivity requests failed"
        [ -z "$low_ok" ] && echo "  Low priority response empty"
        [ -z "$high_ok" ] && echo "  High priority response empty"
    fi

    echo "  Checking container logs for priority_jump..."
    local log_match
    log_match=$(timeout 5 docker logs --tail 200 "$CONTAINER_NAME" 2>&1 | grep -i "priority_jump\|queue.*priority\|latency_sensitivity" | tail -3 || true)
    if [ -n "$log_match" ]; then
        pass "Found priority_jump log entries"
        echo "  $log_match"
    else
        skip "No priority_jump logs found (may need DYNAMO_ROUTER_QUEUE_THRESHOLD set or RUST_LOG=debug)"
    fi

    rm -rf "$tmpdir"
}

# ============================================================================
# Test 3: priority (engine scheduling)
# ============================================================================
test_priority() {
    echo ""
    echo "=== Test 3: priority (Engine Scheduling) ==="

    echo "  Sending request with nvext.agent_hints.priority=1 (high priority)"
    local resp_high
    resp_high=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 5,
        "nvext": {"agent_hints": {"priority": 1}}
    }')
    if [ $? -ne 0 ] || [ -z "$resp_high" ]; then
        fail "High priority request failed"
        return
    fi
    local content_high
    content_high=$(echo "$resp_high" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content_high" ]; then
        pass "High priority (1) request accepted"
    else
        fail "High priority request returned no content"
    fi

    echo "  Sending request with nvext.agent_hints.priority=999 (low priority)"
    local resp_low
    resp_low=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say bye."}],
        "max_tokens": 5,
        "nvext": {"agent_hints": {"priority": 999}}
    }')
    if [ $? -ne 0 ] || [ -z "$resp_low" ]; then
        fail "Low priority request failed"
        return
    fi
    local content_low
    content_low=$(echo "$resp_low" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content_low" ]; then
        pass "Low priority (999) request accepted"
    else
        fail "Low priority request returned no content"
    fi

    echo "  Checking container logs for priority forwarding..."
    local log_match
    log_match=$(timeout 5 docker logs --tail 200 "$CONTAINER_NAME" 2>&1 | grep -i "priority" | grep -v "priority_jump" | tail -3 || true)
    if [ -n "$log_match" ]; then
        pass "Found priority-related log entries"
        echo "  $log_match"
    else
        skip "No priority logs found (may need RUST_LOG=debug)"
    fi
}

# ============================================================================
# Test 4: cache_control (pin_prefix after generation)
# ============================================================================
test_cache_control() {
    echo ""
    echo "=== Test 4: cache_control (Pin Prefix) ==="

    echo "  Test 4a: Sending request with cache_control ttl=5m"
    local resp
    resp=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Explain KV caching in one sentence."}],
        "max_tokens": 50,
        "nvext": {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    }')
    if [ $? -ne 0 ] || [ -z "$resp" ]; then
        fail "cache_control (5m) request failed"
        return
    fi
    local content
    content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content" ]; then
        pass "cache_control (5m) request accepted"
    else
        fail "cache_control (5m) request returned no content"
    fi

    echo "  Test 4b: Sending request without cache_control (should NOT pin)"
    local resp_no_cc
    resp_no_cc=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 5
    }')
    if [ -n "$(echo "$resp_no_cc" | jq -r '.choices[0].message.content // empty' 2>/dev/null)" ]; then
        pass "No-cache_control request accepted (no pin expected)"
    else
        fail "No-cache_control request failed"
    fi

    echo "  Checking container logs for pin_prefix..."
    local log_match
    log_match=$(timeout 5 docker logs --tail 300 "$CONTAINER_NAME" 2>&1 | grep -i "pin_prefix\|cache_control\|PinState\|spawn_pin" | tail -5 || true)
    if [ -n "$log_match" ]; then
        pass "Found cache_control/pin_prefix log entries"
        echo "  $log_match"
    else
        skip "No pin_prefix logs found (may need DYNAMO_ENABLE_CACHE_CONTROL=true and RUST_LOG=debug)"
    fi

    echo "  Test 4c: Re-sending same prompt to test cache hit"
    local start_time end_time ttft_ms
    start_time=$(date +%s%N)
    local resp_reuse
    resp_reuse=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Explain KV caching in one sentence."}],
        "max_tokens": 50,
        "nvext": {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    }')
    end_time=$(date +%s%N)
    if [ -n "$(echo "$resp_reuse" | jq -r '.choices[0].message.content // empty' 2>/dev/null)" ]; then
        ttft_ms=$(( (end_time - start_time) / 1000000 ))
        pass "Cache reuse request completed in ${ttft_ms}ms (compare with first request)"
    else
        fail "Cache reuse request failed"
    fi
}

# ============================================================================
# Test 5: All hints combined
# ============================================================================
test_combined() {
    echo ""
    echo "=== Test 5: All Hints Combined ==="
    echo "  Sending request with osl + latency_sensitivity + priority + cache_control"

    local resp
    resp=$(send_request '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "What is prefix caching?"}],
        "max_tokens": 50,
        "nvext": {
            "agent_hints": {
                "osl": 256,
                "latency_sensitivity": 5.0,
                "priority": 1
            },
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        }
    }')
    if [ $? -ne 0 ] || [ -z "$resp" ]; then
        fail "Combined hints request failed"
        return
    fi
    local content
    content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
    if [ -n "$content" ]; then
        pass "Combined hints request accepted: $(echo "$content" | head -c 50)"
    else
        fail "Combined hints request returned no content"
        echo "  Response: $(echo "$resp" | head -c 200)"
    fi
}

# ============================================================================
# Main
# ============================================================================

echo "=========================================="
echo "Dynamo NvExt Feature Isolation Tests"
echo "=========================================="
echo "API: $API_BASE"
echo "Model: $MODEL_NAME"
echo "Container: $CONTAINER_NAME"
echo ""

check_api

RUN_TEST="${1:-all}"

case "$RUN_TEST" in
    all)
        test_baseline
        test_osl
        test_latency_sensitivity
        test_priority
        test_cache_control
        test_combined
        ;;
    baseline)    test_baseline ;;
    osl)         test_osl ;;
    latency*)    test_latency_sensitivity ;;
    priority)    test_priority ;;
    cache*)      test_cache_control ;;
    combined)    test_combined ;;
    *)
        echo "Unknown test: $RUN_TEST"
        echo "Available: all, baseline, osl, latency_sensitivity, priority, cache_control, combined"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "=========================================="

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Tip: Enable debug logging for more detail:"
    echo "  docker exec $CONTAINER_NAME bash -c 'export RUST_LOG=debug'"
    echo "  docker logs -f $CONTAINER_NAME 2>&1 | grep -i 'nvext\|hint\|priority\|osl\|cache_control\|pin'"
    exit 1
fi
