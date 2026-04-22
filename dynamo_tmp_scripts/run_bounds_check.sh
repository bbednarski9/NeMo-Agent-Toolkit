#!/bin/bash
# Bounds Check — 6 experiments, one variable at a time from Step 2 baseline
# Step 2 baseline: w_p=1.0, w_d=1.0, w_mem=0, ε=0.1, T=0, λ_s=0 → 76.36 TPS

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
MGMT_URL="http://localhost:8084"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/bounds_check_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate

mkdir -p "$OUTDIR"

# Common overrides shared by all experiments
COMMON=(
    --override eval.general.max_concurrency 20
    --override llms.dynamo_llm.router_w_mem 0.0
    --override llms.dynamo_llm.router_lambda_stickiness 0.0
)

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
    curl -s "$MGMT_URL/decisions/summary" | python3 -m json.tool > "$outfile" 2>/dev/null || echo "{}" > "$outfile"
    # Extract key metrics
    python3 -c "
import json
d = json.load(open('$outfile'))
dist = d.get('worker_distribution', {})
counts = list(dist.values())
n_active = len([c for c in counts if c > 0])
total = sum(counts)
evenness = min(counts)/max(counts) if counts else 0
print(f'  Workers: {n_active}/8  Evenness: {evenness:.2f}  Stickiness: {d.get(\"prefix_stickiness_rate\",0):.1%}  Overlap: {d.get(\"avg_kv_overlap_chosen_worker\",0):.3f}  Agreement: {d.get(\"agreement_rate\",0):.1%}')
" 2>/dev/null
}

run_eval() {
    local exp_name=$1
    shift
    local logfile="$OUTDIR/${exp_name}_log_${TIMESTAMP}.txt"
    echo "  Log: $logfile"
    nat eval "$@" > "$logfile" 2>&1
    # Extract TPS/TTFT
    python3 -c "
import re
text = open('$logfile').read()
tps = re.search(r'avg_tps\s+\|\s+([\d.]+)', text)
ttft = re.search(r'avg_ttft\s+\|\s+([\d.]+)', text)
runtime = re.search(r'Total Runtime: ([\d.]+)s', text)
print(f'  TPS: {tps.group(1) if tps else \"?\"}  TTFT: {ttft.group(1) if ttft else \"?\"}  Runtime: {runtime.group(1) if runtime else \"?\"}s')
" 2>/dev/null
}

wait_for_backend

echo "============================================================"
echo "BOUNDS CHECK (baseline: w_p=1, w_d=1, ε=0.1, T=0 → 76.36 TPS)"
echo "============================================================"
echo ""

# --- D1: Cache-heavy ratio (w_prefill=2.0) ---
echo "D1: w_prefill=2.0 (cache-heavy ratio)"
run_eval "D1_cache_heavy" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 2.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.1 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/D1_cache_heavy/"
capture_summary "D1_cache_heavy"
echo ""

# --- D2: Load-heavy ratio (w_decode=2.0) ---
echo "D2: w_decode=2.0 (load-heavy ratio)"
run_eval "D2_load_heavy" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 2.0 \
    --override llms.dynamo_llm.router_epsilon 0.1 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/D2_load_heavy/"
capture_summary "D2_load_heavy"
echo ""

# --- D3: No exploration (ε=0) ---
echo "D3: epsilon=0.0 (no exploration, should match Step 0: ~71)"
run_eval "D3_no_epsilon" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.0 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/D3_no_epsilon/"
capture_summary "D3_no_epsilon"
echo ""

# --- D4: More exploration (ε=0.2) ---
echo "D4: epsilon=0.2 (more exploration)"
run_eval "D4_high_epsilon" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.2 \
    --override llms.dynamo_llm.router_temperature 0.0 \
    --override eval.general.output.dir "$OUTDIR/D4_high_epsilon/"
capture_summary "D4_high_epsilon"
echo ""

# --- D5: Softmax only (T=0.2, ε=0) ---
echo "D5: temperature=0.2, epsilon=0.0 (softmax exploration only)"
run_eval "D5_temp_only" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.0 \
    --override llms.dynamo_llm.router_temperature 0.2 \
    --override eval.general.output.dir "$OUTDIR/D5_temp_only/"
capture_summary "D5_temp_only"
echo ""

# --- D6: Both exploration (T=0.2, ε=0.1) ---
echo "D6: temperature=0.2, epsilon=0.1 (both mechanisms)"
run_eval "D6_both_explore" --config_file "$CONFIG" "${COMMON[@]}" \
    --override llms.dynamo_llm.router_w_prefill 1.0 \
    --override llms.dynamo_llm.router_w_decode 1.0 \
    --override llms.dynamo_llm.router_epsilon 0.1 \
    --override llms.dynamo_llm.router_temperature 0.2 \
    --override eval.general.output.dir "$OUTDIR/D6_both_explore/"
capture_summary "D6_both_explore"
echo ""

echo "============================================================"
echo "BOUNDS CHECK COMPLETE"
echo "============================================================"
echo ""
echo "Results in: $OUTDIR"
echo ""
echo "| Exp | Config Change | TPS | TTFT | Workers | Evenness |"
echo "|-----|--------------|-----|------|---------|----------|"
for exp in D1_cache_heavy D2_load_heavy D3_no_epsilon D4_high_epsilon D5_temp_only D6_both_explore; do
    logfile="$OUTDIR/${exp}_log_${TIMESTAMP}.txt"
    sumfile="$OUTDIR/${exp}_summary_${TIMESTAMP}.json"
    python3 -c "
import re, json
text = open('$logfile').read()
tps = re.search(r'avg_tps\s+\|\s+([\d.]+)', text)
ttft = re.search(r'avg_ttft\s+\|\s+([\d.]+)', text)
d = json.load(open('$sumfile'))
dist = d.get('worker_distribution', {})
counts = list(dist.values())
n = len([c for c in counts if c > 0])
ev = min(counts)/max(counts) if counts else 0
print(f'| $exp | ... | {tps.group(1) if tps else \"?\"} | {ttft.group(1) if ttft else \"?\"} | {n}/8 | {ev:.2f} |')
" 2>/dev/null
done
