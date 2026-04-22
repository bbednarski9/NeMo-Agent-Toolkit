#!/bin/bash
# Layered Feature Evaluation — Step 0 through Step 3
# Each step adds one feature on top of the native parity baseline.
# All at concurrency 20, 500 items.

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/layered_v1"
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
    grep "EVALUATION SUMMARY" -A 12 "$logfile" || tail -10 "$logfile"
    return $rc
}

wait_for_backend

# =========================================================================
# Step 0: Normalization Verification
# Same as Exp A (native parity) but with per-decision normalization active.
# Should match Exp A's 72.24 TPS since normalization preserves argmin ordering.
# =========================================================================
echo "============================================================"
echo "STEP 0: Normalization Verification (native parity + norm)"
echo "============================================================"
run_eval "step0_norm_verify" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_w_mem 0.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.0 \
    --override llms.dynamo_llm.router_epsilon 0.0 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/step0_norm_verify/"
capture_summary "step0_norm_verify"

# =========================================================================
# Step 2: Beta Learner (epsilon=0.1)
# Normalized scores mean epsilon=0.1 = "10% of best-vs-worst gap."
# Should maintain ~72 TPS while exploring worker quality.
# =========================================================================
echo ""
echo "============================================================"
echo "STEP 2: Beta Learner (epsilon=0.1)"
echo "============================================================"
run_eval "step2_beta_learner" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_w_mem 0.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.0 \
    --override llms.dynamo_llm.router_epsilon 0.1 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/step2_beta_learner/"
capture_summary "step2_beta_learner"

# =========================================================================
# Step 3: Stickiness (λ_stickiness=0.5, moderate)
# With normalized scores, stickiness_benefit ~0-1.3 is now meaningful.
# λ=0.5 means stickiness can offset ~50% of the best-vs-worst gap.
# =========================================================================
echo ""
echo "============================================================"
echo "STEP 3: Stickiness (λ_stickiness=0.5, epsilon=0.1)"
echo "============================================================"
run_eval "step3_stickiness" \
    --config_file "$CONFIG" \
    --override eval.general.max_concurrency 20 \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_w_mem 0.0 \
    --override llms.dynamo_llm.router_lambda_stickiness 0.5 \
    --override llms.dynamo_llm.router_epsilon 0.1 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/step3_stickiness/"
capture_summary "step3_stickiness"

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================================"
echo "LAYERED EVALUATION COMPLETE"
echo "============================================================"
echo "Results in: $OUTDIR"
echo ""
echo "Expected progression:"
echo "  Step 0 (norm verify): ~72 TPS (match Exp A)"
echo "  Step 2 (beta):        ~72 TPS (exploration shouldn't hurt)"
echo "  Step 3 (stickiness):  ~68-72 TPS (tradeoff: TPS vs TTFT/cache)"
echo ""
ls -la "$OUTDIR"/*_${TIMESTAMP}.* 2>/dev/null
