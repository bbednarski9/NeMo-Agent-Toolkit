#!/bin/bash
# Outer Bounds Check — 10 experiments, one param at a time
# Center: w_p=1, w_d=1, ε=0.1, λ_s=0.05, λ_l=0.05, alpha=0.5
# Tests high/low for: lambda_stickiness, lambda_lints, alpha_reuse, lints_v, lints_forget_rate

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/outer_bounds_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate
mkdir -p "$OUTDIR"

# Center config (current best: 78.28 TPS)
CENTER=(
    --override eval.general.max_concurrency 20
    --override llms.dynamo_llm.router_w_prefill 1.0
    --override llms.dynamo_llm.router_w_decode 1.0
    --override llms.dynamo_llm.router_w_mem 0.0
    --override llms.dynamo_llm.router_epsilon 0.1
    --override llms.dynamo_llm.router_temperature 0.0
    --override llms.dynamo_llm.router_alpha_reuse 0.5
    --override llms.dynamo_llm.router_lambda_stickiness 0.05
    --override llms.dynamo_llm.router_lambda_lints 0.05
)

wait_for_backend() {
    for i in $(seq 1 60); do
        curl -s http://localhost:8000/v1/models | grep -q "Llama" && return 0
        sleep 5
    done
    return 1
}

run_and_report() {
    local name=$1; shift
    local logfile="$OUTDIR/${name}_log_${TIMESTAMP}.txt"
    local sumfile="$OUTDIR/${name}_summary_${TIMESTAMP}.json"
    echo -n "  $name: "
    nat eval "$@" > "$logfile" 2>&1
    curl -s "$MGMT_URL/decisions/summary" | python3 -m json.tool > "$sumfile" 2>/dev/null
    python3 -c "
import re, json
text = open('$logfile').read()
tps = re.search(r'avg_tps\s+\|\s+([\d.]+)', text)
ttft = re.search(r'avg_ttft\s+\|\s+([\d.]+)', text)
d = json.load(open('$sumfile'))
dist = d.get('worker_distribution', {})
counts = list(dist.values())
ev = min(counts)/max(counts) if counts else 0
print(f'TPS={tps.group(1) if tps else \"?\":>6s}  TTFT={ttft.group(1) if ttft else \"?\":>5s}  Even={ev:.2f}  Overlap={d.get(\"avg_kv_overlap_chosen_worker\",0):.3f}')
" 2>/dev/null
}

wait_for_backend
echo "============================================================"
echo "OUTER BOUNDS CHECK (center: 78.28 TPS)"
echo "One param at a time, high/low from center"
echo "============================================================"
echo ""

# --- lambda_stickiness: center=0.05, test [0.01, 0.15] ---
echo "lambda_stickiness:"
run_and_report "ls_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_stickiness 0.01 \
    --override eval.general.output.dir "$OUTDIR/ls_low/"
run_and_report "ls_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_stickiness 0.15 \
    --override eval.general.output.dir "$OUTDIR/ls_high/"
echo ""

# --- lambda_lints: center=0.05, test [0.01, 0.15] ---
echo "lambda_lints:"
run_and_report "ll_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_lints 0.01 \
    --override eval.general.output.dir "$OUTDIR/ll_low/"
run_and_report "ll_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_lints 0.15 \
    --override eval.general.output.dir "$OUTDIR/ll_high/"
echo ""

# --- alpha_reuse: center=0.5, test [0.25, 0.75] ---
echo "alpha_reuse:"
run_and_report "ar_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.25 \
    --override eval.general.output.dir "$OUTDIR/ar_low/"
run_and_report "ar_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.75 \
    --override eval.general.output.dir "$OUTDIR/ar_high/"
echo ""

# --- lints_v: center=0.25, test [0.1, 0.5] ---
echo "lints_v:"
run_and_report "lv_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lints_v 0.1 \
    --override eval.general.output.dir "$OUTDIR/lv_low/"
run_and_report "lv_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lints_v 0.5 \
    --override eval.general.output.dir "$OUTDIR/lv_high/"
echo ""

# --- lints_forget_rate: center=0.995, test [0.99, 0.999] ---
echo "lints_forget_rate:"
run_and_report "lf_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lints_forget_rate 0.99 \
    --override eval.general.output.dir "$OUTDIR/lf_low/"
run_and_report "lf_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lints_forget_rate 0.999 \
    --override eval.general.output.dir "$OUTDIR/lf_high/"

# ============================================================
# Interaction cases — key param pairs at their bounds
# ============================================================
echo "--- Interaction Cases ---"
echo ""

# I1: Both learners maxed (high epsilon + high lints)
echo "Interactions:"
run_and_report "I1_both_learners_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_epsilon 0.15 \
    --override llms.dynamo_llm.router_lambda_lints 0.15 \
    --override eval.general.output.dir "$OUTDIR/I1_both_learners_high/"

# I2: Both learners minimal (low epsilon + low lints)
run_and_report "I2_both_learners_low" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_epsilon 0.05 \
    --override llms.dynamo_llm.router_lambda_lints 0.01 \
    --override eval.general.output.dir "$OUTDIR/I2_both_learners_low/"

# I3: High stickiness + high lints (both additive corrections maxed)
run_and_report "I3_sticky_lints_high" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_stickiness 0.15 \
    --override llms.dynamo_llm.router_lambda_lints 0.15 \
    --override eval.general.output.dir "$OUTDIR/I3_sticky_lints_high/"

# I4: Low stickiness + high lints (lints dominates corrections)
run_and_report "I4_low_sticky_high_lints" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_lambda_stickiness 0.01 \
    --override llms.dynamo_llm.router_lambda_lints 0.15 \
    --override eval.general.output.dir "$OUTDIR/I4_low_sticky_high_lints/"

# I5: High alpha_reuse + high lints (aggressive cache weighting + aggressive learning)
run_and_report "I5_high_reuse_high_lints" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.75 \
    --override llms.dynamo_llm.router_lambda_lints 0.15 \
    --override eval.general.output.dir "$OUTDIR/I5_high_reuse_high_lints/"

# I6: Low alpha_reuse + low lints (minimal future-awareness + minimal learning)
run_and_report "I6_low_reuse_low_lints" --config_file "$CONFIG" "${CENTER[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.25 \
    --override llms.dynamo_llm.router_lambda_lints 0.01 \
    --override eval.general.output.dir "$OUTDIR/I6_low_reuse_low_lints/"

echo ""
echo "============================================================"
echo "OUTER BOUNDS + INTERACTIONS COMPLETE"
echo "============================================================"
echo ""
echo "Summary (center=78.28 TPS):"
echo "One-at-a-time bounds + 6 interaction cases = 16 experiments"
