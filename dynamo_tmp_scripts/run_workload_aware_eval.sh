#!/bin/bash
# Step 1: Future Workload Aware Ranking
# Tests alpha_reuse modulation of w_prefill
# Baseline: w_p=1.0, w_d=1.0, ε=0.1, T=0, λ_s=0, alpha_reuse=0.25 → 76.36 TPS

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/workload_aware_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate
mkdir -p "$OUTDIR"

COMMON=(
    --override eval.general.max_concurrency 20
    --override llms.dynamo_llm.router_w_prefill 1.0
    --override llms.dynamo_llm.router_w_decode 1.0
    --override llms.dynamo_llm.router_w_mem 0.0
    --override llms.dynamo_llm.router_lambda_stickiness 0.0
    --override llms.dynamo_llm.router_epsilon 0.1
    --override llms.dynamo_llm.router_temperature 0.0
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
    echo "  Running $name..."
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
print(f'  {\"$name\":30s} TPS={tps.group(1) if tps else \"?\":>6s}  TTFT={ttft.group(1) if ttft else \"?\":>5s}  Even={ev:.2f}  Overlap={d.get(\"avg_kv_overlap_chosen_worker\",0):.3f}  Sticky={d.get(\"prefix_stickiness_rate\",0):.1%}')
" 2>/dev/null
}

wait_for_backend
echo "============================================================"
echo "WORKLOAD-AWARE RANKING (alpha_reuse modulates w_prefill)"
echo "Baseline: w_p=1.0, w_d=1.0, ε=0.1 → 76.36 TPS"
echo "============================================================"
echo ""

# W1: alpha_reuse=0.0 (disabled — should match baseline)
echo "W1: alpha_reuse=0.0 (disabled, baseline verification)"
run_and_report "W1_reuse_0" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.0 \
    --override eval.general.output.dir "$OUTDIR/W1_reuse_0/"

# W2: alpha_reuse=0.25 (default — modest boost)
echo "W2: alpha_reuse=0.25 (default, modest boost)"
run_and_report "W2_reuse_025" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.25 \
    --override eval.general.output.dir "$OUTDIR/W2_reuse_025/"

# W3: alpha_reuse=0.5 (strong boost — first request gets w_prefill*1.5)
echo "W3: alpha_reuse=0.5 (strong boost)"
run_and_report "W3_reuse_050" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.5 \
    --override eval.general.output.dir "$OUTDIR/W3_reuse_050/"

# W4: alpha_reuse=0.75 (aggressive — first request gets w_prefill*1.75)
echo "W4: alpha_reuse=0.75 (aggressive boost)"
run_and_report "W4_reuse_075" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_alpha_reuse 0.75 \
    --override eval.general.output.dir "$OUTDIR/W4_reuse_075/"

echo ""
echo "============================================================"
echo "WORKLOAD-AWARE EVAL COMPLETE"
echo "============================================================"
