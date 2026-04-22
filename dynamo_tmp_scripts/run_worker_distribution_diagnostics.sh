#!/bin/bash
# Worker Distribution Diagnostics — 4 experiments at concurrency 20
# Uses eval_trial10_concurrency.yml as base config with --override flags

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
DATASET_FULL="examples/dynamo_integration/data/agent_leaderboard_v2_all.json"
DATASET_50="examples/dynamo_integration/data/agent_leaderboard_v2_all_50.json"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/two_term_v1/worker_diag"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate

mkdir -p "$OUTDIR"

# Skip Exp 1 — already completed. Set SKIP_EXP1=false to re-run.
SKIP_EXP1=${SKIP_EXP1:-true}

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

reset_router() {
    echo "Resetting router stats..."
    curl -s "$MGMT_URL/decisions/reset" -X POST 2>/dev/null || echo "(reset endpoint not available)"
    sleep 2
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
    # Use script redirect instead of tee to avoid SIGPIPE
    nat eval "$@" > "$logfile" 2>&1
    local rc=$?
    # Show summary from log
    tail -20 "$logfile"
    return $rc
}

wait_for_backend

# =========================================================================
# Experiment 1: Diagnostic Logging (50 items, concurrency 20)
# =========================================================================
if [ "$SKIP_EXP1" = "false" ]; then
    echo "============================================================"
    echo "EXPERIMENT 1: Diagnostic Logging (50 items, conc=20)"
    echo "============================================================"
    reset_router
    run_eval "exp1_diagnostic" \
        --config_file "$CONFIG" \
        --dataset "$DATASET_50" \
        --override eval.general.max_concurrency 20 \
        --override eval.general.output.dir "$OUTDIR/exp1_diagnostic/"
    capture_summary "exp1_diagnostic"
    echo "--- Diagnostic log lines ---"
    docker logs dynamo-vllm 2>&1 | grep -E "DIAG wid=|LOAD_FALLBACK" | head -50 > "$OUTDIR/exp1_diag_lines_${TIMESTAMP}.txt"
    cat "$OUTDIR/exp1_diag_lines_${TIMESTAMP}.txt"
else
    echo "SKIPPING Experiment 1 (already completed)"
fi

# =========================================================================
# Experiment 2: Kill Stickiness (lambda_stick=0)
# =========================================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 2: Kill Stickiness (lambda_stickiness=0.0)"
echo "============================================================"
reset_router
run_eval "exp2_kill_stickiness" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.0 \
    --override eval.general.output.dir "$OUTDIR/exp2_kill_stickiness/"
capture_summary "exp2_kill_stickiness"

# =========================================================================
# Experiment 3: Kill Reuse Budget (total_requests=1)
# =========================================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 3: Kill Reuse Budget (total_requests=1)"
echo "============================================================"
reset_router
run_eval "exp3_kill_reuse_budget" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.nvext_prefix_total_requests 1 \
    --override eval.general.output.dir "$OUTDIR/exp3_kill_reuse_budget/"
capture_summary "exp3_kill_reuse_budget"

# =========================================================================
# Experiment 4: Extreme Load Discount (w_osl_load=50, lambda_stick=0.5)
# =========================================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 4: Extreme Load Discount (w_osl_load=50)"
echo "============================================================"
reset_router
run_eval "exp4_extreme_load" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_osl_load 50.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.5 \
    --override eval.general.output.dir "$OUTDIR/exp4_extreme_load/"
capture_summary "exp4_extreme_load"

# =========================================================================
# Final Summary
# =========================================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results in: $OUTDIR"
echo ""
echo "Decision tree:"
echo "  Exp 1: Check exp1_diag_lines -- kv_util zeros? tiny? fallback?"
echo "  Exp 2: Check exp2 summary -- workers spread to 6+?"
echo "  Exp 3: Check exp3 summary -- workers spread to 6+?"
echo "  Exp 4: Check exp4 summary -- workers spread to 6+?"
echo ""
ls -la "$OUTDIR"/*_${TIMESTAMP}.* 2>/dev/null
