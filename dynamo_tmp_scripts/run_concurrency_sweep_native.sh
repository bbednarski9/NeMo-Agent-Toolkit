#!/bin/bash
# Concurrency Sweep — KV-native router baseline

CONFIG="examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/eval_trial10_concurrency.yml"
OUTDIR="examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/concurrency_sweep_native_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd /localhome/local-bbednarski/NeMo-Agent-Toolkit
source /localhome/local-bbednarski/.venvs/nat_dynamo_eval/bin/activate
mkdir -p "$OUTDIR"

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
    echo -n "  $name: "
    nat eval "$@" > "$logfile" 2>&1
    python3 -c "
import re
text = open('$logfile').read()
tps = re.search(r'avg_tps\s+\|\s+([\d.]+)', text)
ttft = re.search(r'avg_ttft\s+\|\s+([\d.]+)', text)
runtime = re.search(r'Total Runtime: ([\d.]+)s', text)
print(f'TPS={tps.group(1) if tps else \"?\":>6s}  TTFT={ttft.group(1) if ttft else \"?\":>5s}  Runtime={runtime.group(1) if runtime else \"?\"}s')
" 2>/dev/null
}

wait_for_backend
echo "============================================================"
echo "CONCURRENCY SWEEP (KV-native router)"
echo "============================================================"
echo ""

for CONC in 20 40 60 80; do
    echo "Concurrency $CONC:"
    run_and_report "conc_${CONC}" --config_file "$CONFIG" \
        --override eval.general.max_concurrency $CONC \
        --override eval.general.output.dir "$OUTDIR/conc_${CONC}/"
    echo ""
done

echo "============================================================"
echo "CONCURRENCY SWEEP (NATIVE) COMPLETE"
echo "============================================================"
