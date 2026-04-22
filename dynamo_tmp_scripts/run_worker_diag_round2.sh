#!/bin/bash
# Worker Distribution Diagnostics — Round 2 (with stickiness decay fix)
# Re-runs baseline (Trial 10 defaults) + Exp 4 (extreme load discount)

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/two_term_v1/worker_diag_round2"
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
    tail -20 "$logfile"
    return $rc
}

wait_for_backend

# =========================================================================
# Baseline: Trial 10 defaults WITH stickiness decay fix
# Compare to Round 1 Exp 1: 2/8 workers, TPS=34.71
# =========================================================================
echo "============================================================"
echo "BASELINE: Trial 10 defaults + stickiness decay fix (conc=20)"
echo "============================================================"
run_eval "baseline_fixed" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override eval.general.output.dir "$OUTDIR/baseline_fixed/"
capture_summary "baseline_fixed"

# Grab diagnostic log lines
echo "--- Diagnostic log lines (first 30) ---"
docker logs dynamo-vllm 2>&1 | grep -E "DIAG wid=|LOAD_FALLBACK" | head -30 > "$OUTDIR/baseline_diag_lines_${TIMESTAMP}.txt"
cat "$OUTDIR/baseline_diag_lines_${TIMESTAMP}.txt"
echo ""

# =========================================================================
# Experiment 4: Extreme Load Discount (w_osl_load=50, lambda_stick=0.5)
# Failed in Round 1 due to schema validation (le=10, now le=100)
# =========================================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 4: Extreme Load Discount (w_osl_load=50)"
echo "============================================================"
run_eval "exp4_extreme_load" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_osl_load 50.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.5 \
    --override eval.general.output.dir "$OUTDIR/exp4_extreme_load/"
capture_summary "exp4_extreme_load"

# =========================================================================
# Final comparison
# =========================================================================
echo ""
echo "============================================================"
echo "ROUND 2 COMPLETE"
echo "============================================================"
echo "Results in: $OUTDIR"
echo ""
echo "Compare to Round 1:"
echo "  Round 1 Baseline: 2/8 workers, TPS=34.71, stickiness=98%"
echo "  Round 1 Exp 2 (no stickiness): 7/8 workers, TPS=48.39"
echo "  Round 1 Exp 3 (no reuse budget): 6/8 workers, TPS=41.39"
echo ""
ls -la "$OUTDIR"/*_${TIMESTAMP}.* 2>/dev/null
