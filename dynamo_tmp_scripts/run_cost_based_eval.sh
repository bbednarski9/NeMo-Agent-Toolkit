#!/bin/bash
# Cost-Based Router Evaluation — Round 3
# Tests native router parity + stickiness layering

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/cost_based_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate

mkdir -p "$OUTDIR"

wait_for_backend() {
    echo "Waiting for backend..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/v1/models | grep -q "Llama"; then
            echo "Backend ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Backend did not come up in 5 minutes"
    return 1
}

capture_summary() {
    local exp_name=$1
    local outfile="$OUTDIR/${exp_name}_summary_${TIMESTAMP}.json"
    echo "Capturing /decisions/summary -> $outfile"
    curl -s "$MGMT_URL/decisions/summary" | python3 -m json.tool > "$outfile" 2>/dev/null || echo "{\"error\": \"summary not available\"}" > "$outfile"
    echo "--- $exp_name summary ---"
    cat "$outfile"
    echo ""
}

run_eval() {
    local exp_name=$1
    shift
    local logfile="$OUTDIR/${exp_name}_log_${TIMESTAMP}.txt"
    echo "Logging to: $logfile"
    nat eval "$@" > "$logfile" 2>&1
    local rc=$?
    # Show eval summary
    grep "EVALUATION SUMMARY" -A 12 "$logfile" || tail -10 "$logfile"
    return $rc
}

wait_for_backend

# =========================================================================
# Exp A: Native Router Parity
# w_prefill=1.0, w_decode=1.0, λ_stickiness=0, w_mem=0, epsilon=0
# Should match KV-native's 72.9 TPS if formula is aligned
# =========================================================================
echo "============================================================"
echo "EXP A: Native Router Parity (w_prefill=1.0, w_decode=1.0, no stickiness)"
echo "============================================================"

# Hot-reload params to match native router exactly
curl -s "$MGMT_URL/config" -X POST -H "Content-Type: application/json" \
    -d '{"w_prefill": 1.0, "w_decode": 1.0, "w_mem": 0.0, "lambda_stickiness": 0.0, "epsilon": 0.0, "temperature": 0.0}' || true
sleep 2

run_eval "expA_native_parity" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_w_mem 0.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.0 \
    --override llms.dynamo_llm.router_epsilon 0.0 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/expA_native_parity/"
capture_summary "expA_native_parity"

# =========================================================================
# Exp B: Native Parity + Stickiness
# Same as A but with stickiness enabled (λ_stickiness=1.0)
# Tests whether stickiness helps or hurts on top of native-aligned scoring
# =========================================================================
echo ""
echo "============================================================"
echo "EXP B: Native Parity + Stickiness (λ_stickiness=1.0)"
echo "============================================================"

run_eval "expB_native_plus_stickiness" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_w_mem 0.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.0 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/expB_native_plus_stickiness/"
capture_summary "expB_native_plus_stickiness"

# =========================================================================
# Exp C: Default Config
# Uses eval_trial10_concurrency.yml defaults (w_prefill=0.55, w_decode=0.30, etc.)
# Baseline for the new cost-based architecture
# =========================================================================
echo ""
echo "============================================================"
echo "EXP C: Default Cost-Based Config"
echo "============================================================"

run_eval "expC_default_config" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override eval.general.output.dir "$OUTDIR/expC_default_config/"
capture_summary "expC_default_config"

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================================"
echo "ROUND 3 COMPLETE"
echo "============================================================"
echo "Results in: $OUTDIR"
echo ""
echo "Compare to baselines:"
echo "  KV-native:        8/8 workers, TPS=72.9, TTFT=0.18"
echo "  Old Trial 10:     2/8 workers, TPS=34.71 (broken stickiness)"
echo "  Stickiness fix:   8/8 workers, TPS=40.95 (old load signals)"
echo ""
ls -la "$OUTDIR"/*_${TIMESTAMP}.* 2>/dev/null
