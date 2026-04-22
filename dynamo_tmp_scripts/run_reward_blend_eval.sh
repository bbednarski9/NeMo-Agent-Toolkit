#!/bin/bash
# Reward Blend Eval — test TTFT-weighted reward functions
# Base: best config (w_p=1, w_d=1, ε=0.1, λ_s=0.05, λ_l=0.05, α=0.5)

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/reward_blend_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate
mkdir -p "$OUTDIR"

COMMON=(
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
echo "REWARD BLEND (0=pure TPS, 1=pure TTFT)"
echo "============================================================"
echo ""

# R0: Pure TPS (current default)
echo "R0: reward_ttft_weight=0.0 (pure TPS, baseline)"
run_and_report "R0_tps" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_reward_ttft_weight 0.0 \
    --override eval.general.output.dir "$OUTDIR/R0_tps/"

# R1: Light TTFT blend
echo "R1: reward_ttft_weight=0.25"
run_and_report "R1_blend_025" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_reward_ttft_weight 0.25 \
    --override eval.general.output.dir "$OUTDIR/R1_blend_025/"

# R2: Even blend
echo "R2: reward_ttft_weight=0.5"
run_and_report "R2_blend_050" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_reward_ttft_weight 0.5 \
    --override eval.general.output.dir "$OUTDIR/R2_blend_050/"

# R3: TTFT-heavy
echo "R3: reward_ttft_weight=0.75"
run_and_report "R3_blend_075" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_reward_ttft_weight 0.75 \
    --override eval.general.output.dir "$OUTDIR/R3_blend_075/"

echo ""
echo "============================================================"
echo "REWARD BLEND EVAL COMPLETE"
echo "============================================================"
