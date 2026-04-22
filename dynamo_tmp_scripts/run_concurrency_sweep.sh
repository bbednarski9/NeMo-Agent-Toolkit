#!/bin/bash
# Concurrency Sweep — test best config at 20, 40, 60, 80 concurrent requests

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/concurrency_sweep_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate
mkdir -p "$OUTDIR"

COMMON=(
    --override llms.dynamo_llm.router_w_prefill 1.0
    --override llms.dynamo_llm.router_w_decode 1.0
    --override llms.dynamo_llm.router_w_mem 0.0
    --override llms.dynamo_llm.router_epsilon 0.1
    --override llms.dynamo_llm.router_temperature 0.0
    --override llms.dynamo_llm.router_alpha_reuse 0.5
    --override llms.dynamo_llm.router_lambda_stickiness 0.05
    --override llms.dynamo_llm.router_lambda_lints 0.05
    --override llms.dynamo_llm.router_reward_ttft_weight 0.0
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
runtime = re.search(r'Total Runtime: ([\d.]+)s', text)
d = json.load(open('$sumfile'))
dist = d.get('worker_distribution', {})
counts = list(dist.values())
ev = min(counts)/max(counts) if counts else 0
print(f'TPS={tps.group(1) if tps else \"?\":>6s}  TTFT={ttft.group(1) if ttft else \"?\":>5s}  Even={ev:.2f}  Overlap={d.get(\"avg_kv_overlap_chosen_worker\",0):.3f}  Runtime={runtime.group(1) if runtime else \"?\"}s')
" 2>/dev/null
}

wait_for_backend
echo "============================================================"
echo "CONCURRENCY SWEEP (Thompson best config)"
echo "============================================================"
echo ""

for CONC in 20 40 60 80; do
    echo "Concurrency $CONC:"
    run_and_report "conc_${CONC}" --config_file "$CONFIG" "${COMMON[@]}" \
        --override eval.general.max_concurrency $CONC \
        --override eval.general.output.dir "$OUTDIR/conc_${CONC}/"
    echo ""
done

echo "============================================================"
echo "CONCURRENCY SWEEP COMPLETE"
echo "============================================================"
