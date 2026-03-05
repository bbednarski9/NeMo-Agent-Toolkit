# Router Pretraining Pipeline

End-to-end process for training and evaluating the Thompson Sampling router
on the Agent Leaderboard v2 multi-domain benchmark.

## Overview

The pipeline has four phases, executed in order:

```
Phase 1: Prediction Trie Generation   (deterministic, ~5 min)
Phase 2: Hyperparameter Optimization  (Bayesian, ~3-6 min per trial)
Phase 3: Learner State Accumulation   (train set, ~5 min)
Phase 4: Held-Out Evaluation          (test set, ~15-30 min)
```

Each phase builds on artifacts from the previous one. Phases can be skipped
with `--skip-*` flags if their artifacts already exist.

## Prerequisites

- Dynamo stack running with KV-aware Thompson router (`kv_thompson_native`)
- Learner management server on port 8084 (`curl http://localhost:8084/health`)
- NAT eval venv activated (`source ~/.venvs/nat_dynamo_eval/bin/activate`)
- Agent Leaderboard v2 datasets downloaded for all 5 domains

## Environment Variables

All variables are set in `external/dynamo/.env` and passed to the Dynamo
container via the startup script. Variables marked **(container)** must be
visible inside the Docker container; variables marked **(host)** are read
by `nat eval`/`nat optimize` on the host.

### Infrastructure

| Variable | Default | Scope | Description |
|---|---|---|---|
| `DYNAMO_KV_BLOCK_SIZE` | `16` | container | KV cache block size in tokens; must match across all components |
| `DYNAMO_GPU_DEVICES` | `0,1,2,3,4,5,6,7` | container | GPU device IDs for unified mode workers |
| `DYNAMO_TP_SIZE` | `1` | container | Tensor parallelism size per worker |
| `DYNAMO_MEM_FRACTION_STATIC` | `0.5` | container | Fraction of GPU memory for KV cache (0.0-1.0) |
| `DYNAMO_NUM_GPU_BLOCKS_OVERRIDE` | `6500` | container | Fixed KV cache blocks per worker (overrides fraction) |
| `DYNAMO_NUM_GPU_BLOCKS_STEP` | `0` | container | Per-worker block increment (worker_i gets base + i*step) |
| `DYNAMO_WORKER_COMPONENT` | `worker` | container | Component name for workers (`worker` for SGLang, `backend` for vLLM) |
| `DYNAMO_FROM_SOURCE` | `true` | host | Use source-built Dynamo image |
| `DYNAMO_IMAGE` | `dynamo-vllm-source:main` | host | Docker image for the Dynamo container, unused if DYNAMO_FROM_SOURCE=false |
| `ENABLE_KV_AWARE_ROUTING` | `true` | container | Enable KV-aware routing in the frontend |
| `DYNAMO_ENABLE_KV_EVENTS` | `true` | container | Enable KV cache event streaming (vLLM only) |
| `LEARNER_STATE_PORT` | `8084` | container | HTTP port for the learner management server |

### Router Tuning (read from `config.yaml` inside container)

| Variable | Default | Scope | Description |
|---|---|---|---|
| `ROUTER_CONFIG_PATH` | `/workspace/custom_dynamo/config.yaml` | container | Path to router config YAML (read-only in container) |

### Hint Overrides (bypass agent hints for testing)

| Variable | Default | Scope | Description |
|---|---|---|---|
| `HINT_OVERRIDE_OSL` | *(unset)* | container | Override output sequence length hint for all requests |
| `HINT_OVERRIDE_IAT` | *(unset)* | container | Override inter-arrival time hint for all requests |
| `HINT_OVERRIDE_TOTAL_REQUESTS` | *(unset)* | container | Override total requests hint for all requests |
| `HINT_OVERRIDE_PREFIX_ID` | *(unset)* | container | Override prefix ID for all requests |

### Replay Logging

| Variable | Default | Scope | Description |
|---|---|---|---|
| `REPLAY_LOG_DIR` | *(unset)* | container | Path inside container for JSONL replay logs. Set to `/workspace/logs/replay` to enable. When unset, replay logging is disabled with zero overhead. |
| `PROCESSOR_OVERHEAD_LOG` | `/tmp/processor_overhead.jsonl` | container | Path for per-request overhead timing log |

### Monitoring

| Variable | Default | Scope | Description |
|---|---|---|---|
| `GF_AUTH_ANONYMOUS_ENABLED` | `true` | host | Allow unauthenticated Grafana access |
| `GF_AUTH_ANONYMOUS_ORG_ROLE` | `Admin` | host | Role for anonymous Grafana users |
| `GF_AUTH_DISABLE_LOGIN_FORM` | `true` | host | Hide Grafana login form |

### NvExt Feature Flags

| Variable | Default | Scope | Description |
|---|---|---|---|
| `DYNAMO_ENABLE_CACHE_CONTROL` | `false` | container | Enable `cache_control` (pin prefix after generation) |
| `DYNAMO_ENABLE_HIERARCHICAL_CACHE` | `false` | container | Enable hierarchical cache on SGLang workers |

## Data Split

The full dataset (100 scenarios per domain, 500 total) is split into:

- **Training set (20%)**: 20 scenarios per domain, used for trie generation,
  optimization, and learner state accumulation
- **Test set (80%)**: 80 scenarios per domain, used for held-out evaluation

Generate the splits:

```bash
cd ~/NeMo-Agent-Toolkit
for domain in banking healthcare insurance investment telecom; do
    python3 examples/dynamo_integration/scripts/split_agent_leaderboard_v2.py \
        --input-file examples/dynamo_integration/data/agent_leaderboard_v2_${domain}.json \
        --split-by-percentages 20 80
done
```

This produces per-domain files:
- `agent_leaderboard_v2_{domain}_split0_pct20of100.json` (train)
- `agent_leaderboard_v2_{domain}_split1_pct80of100.json` (test)

The splits use a deterministic shuffle seeded by the filename, so repeated
runs produce identical partitions.

## Config Directory

All configs live in:

```
react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/
  {domain}_train.yml        # Train: builds prediction trie + learner state
  {domain}_eval.yml         # Eval: held-out test, consumes trie
  optimize_all_v0.yml       # Optimizer: Bayesian search over 7 router params
  tools/                    # Per-domain tool definitions
```

---

## Phase 1: Prediction Trie Generation

The prediction trie captures per-call-position statistics (remaining calls,
inter-arrival time, output sequence length, latency sensitivity) from the
training set. These statistics replace the static `nvext_prefix_*` hints
with accurate per-call predictions during routing.

### Relevant Environment Variables

| Variable | Purpose in this phase |
|---|---|
| `DYNAMO_KV_BLOCK_SIZE` | Affects KV cache overlap computation during trie profiling |
| `DYNAMO_NUM_GPU_BLOCKS_OVERRIDE` | Worker cache capacity affects routing decisions captured in trie |
| `REPLAY_LOG_DIR` | If set, routing decisions during trie generation are logged |

### Run

```bash
bash examples/dynamo_integration/scripts/run_multi_domain_benchmark.sh \
    --configs examples/dynamo_integration/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/*_train.yml
```

This runs all 5 `{domain}_train.yml` configs concurrently. Each produces
a `prediction_trie.json` in its job output directory.

### Collect

```bash
bash examples/dynamo_integration/scripts/collect_prediction_tries.sh \
    --output-base examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain
```

This copies the per-domain tries into a central directory and builds a
merged all-domain trie:

```
data/prediction_tries/
  banking_prediction_trie.json
  healthcare_prediction_trie.json
  insurance_prediction_trie.json
  investment_prediction_trie.json
  telecom_prediction_trie.json
  all_prediction_trie.json          # merged, used by optimizer
```

The trie keys on the workflow function call path (e.g. `["react_agent"]`),
not on domain content. A single merged trie works across all domains since
they share the same ReAct agent workflow structure.

### Verify

```bash
ls -lh examples/dynamo_integration/data/prediction_tries/
```

All 6 files should exist (5 per-domain + 1 merged).

---

## Phase 2: Hyperparameter Optimization

The optimizer runs Bayesian search (Optuna TPE/NSGA-II) over 7 Thompson
Sampling router parameters, using TTFT, TPS, and ITL as multi-objective
metrics.

### Relevant Environment Variables

| Variable | Purpose in this phase |
|---|---|
| `LEARNER_STATE_PORT` | Port for the management server (optimizer calls `/state/reset` and `/config` per trial) |
| `ROUTER_CONFIG_PATH` | Where `/config` POST writes params (fails if read-only) |
| `HINT_OVERRIDE_*` | If set, overrides trie-derived hints — should be unset during optimization |
| `REPLAY_LOG_DIR` | If set, captures per-trial routing decisions for post-hoc analysis |

### Parameters and Search Spaces

| Parameter | Default | Search Range | Description |
|---|---|---|---|
| `router_ts_weight` | 0.05 | [0.01, 0.20] | Beta-TS exploration weight |
| `router_temperature` | 0.30 | [0.10, 2.00] | Softmax temperature for worker selection |
| `router_cold_start_threshold` | 0.05 | [0.01, 0.40] | Min KV overlap to trust scoring |
| `router_idle_boost` | 0.02 | [0.005, 0.20] | Floor overlap for idle workers |
| `router_beta_decay` | 0.995 | [0.950, 1.000] | Beta learner exponential decay |
| `router_lints_v` | 0.25 | [0.01, 1.00] | LinTS posterior exploration variance |
| `router_lints_forget_rate` | 0.995 | [0.950, 0.999] | LinTS exponential forgetting rate |

### Run

```bash
nat optimize \
    --config_file examples/dynamo_integration/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/optimize_all_v0.yml
```

The optimizer:
1. Resets the learner state before each trial (`POST /state/reset`)
2. Pushes trial parameters to the router (`POST /config`)
3. Runs `nat eval` on the merged training dataset (100 scenarios)
4. Scores the trial on TTFT (40%), TPS (40%), ITL (20%)

### Configuration

In `optimize_all_v0.yml`:
- `optimizer.numeric.n_trials`: Number of Bayesian trials (default: 2, increase for better results)
- `optimizer.numeric.sampler`: `bayesian` (TPE for single-objective, NSGA-II for multi)
- `eval.general.max_concurrency`: Scenarios evaluated in parallel (default: 4)

### Outputs

```
pretrain_optimization/all_split0/optimizer_results/
  trials_dataframe_params.csv     # All trials with params and scores
  optimized_config.yml            # Best parameter set as a config
  plots/                          # Pareto front visualizations
```

### Verify

```bash
cat examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/all_split0/optimizer_results/trials_dataframe_params.csv
```

Check that `values_ttft`, `values_tps`, `values_itl` are nonzero and that
`state=COMPLETE` for all trials.

---

## Phase 3: Learner State Accumulation

After optimization finds the best hyperparameters, we reset the learner
and re-run the training set to accumulate a full learner state (Beta and
LinTS weights per worker) with the optimized params.

### Relevant Environment Variables

| Variable | Purpose in this phase |
|---|---|
| `LEARNER_STATE_PORT` | Port for saving state via `GET /state` |
| `DYNAMO_NUM_GPU_BLOCKS_OVERRIDE` | Worker cache capacity affects routing rewards and learner evolution |
| `REPLAY_LOG_DIR` | If set, captures the full learning trajectory for visualization |

### Run

```bash
# 1. Reset to pristine
curl -sf -X POST http://localhost:8084/state/reset

# 2. Run training set (builds learner state + prediction tries)
bash examples/dynamo_integration/scripts/run_multi_domain_benchmark.sh \
    --configs examples/dynamo_integration/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/*_train.yml

# 3. Save the accumulated state
curl -sf http://localhost:8084/state -o \
    examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/learner_state.json
```

### What the learner state contains

- **BetaLearner**: Per-worker `(alpha, beta)` parameters for Thompson
  Sampling. Workers that consistently deliver lower latency accumulate
  higher alpha (reward signal).
- **LinTSLearner**: Per-worker `(A, b)` matrices for the contextual bandit.
  The 6-dimensional feature vector captures `[bias, inv_prefill, inv_decode,
  affinity, osl_norm, reuse_norm]` — the learner discovers which features
  predict good routing outcomes per worker.

### Verify

```bash
curl -sf http://localhost:8084/state | python3 -c "
import json, sys
d = json.load(sys.stdin)
beta = d['beta_learner']
lints = d['lints_learner']
print(f'Beta: {len(beta[\"bandits\"])} workers, decay={beta[\"decay\"]}')
print(f'LinTS: {len(lints[\"workers\"])} workers, v={lints[\"v\"]}, forget={lints[\"forget_rate\"]}')
"
```

---

## Phase 4: Held-Out Evaluation

Evaluate on the 80% test set using the pretrained learner state and
domain-specific prediction tries.

### Relevant Environment Variables

| Variable | Purpose in this phase |
|---|---|
| `LEARNER_STATE_PORT` | Port for loading pretrained state via `POST /state` |
| `REPLAY_LOG_DIR` | If set, captures held-out routing decisions (keep separate from training logs by clearing the directory or using a different path) |
| `HINT_OVERRIDE_*` | Must be unset — eval should use trie-derived hints |
| `DYNAMO_NUM_GPU_BLOCKS_OVERRIDE` | Should match training to ensure comparable routing behavior |

### Run

```bash
# 1. Reset and load pretrained state
curl -sf -X POST http://localhost:8084/state/reset
curl -sf -X POST http://localhost:8084/state \
    -H 'Content-Type: application/json' \
    -d @examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/pretrain_optimization/learner_state.json

# 2. Run held-out evaluation (5 domains × 80 scenarios = 400 total)
bash examples/dynamo_integration/scripts/run_multi_domain_benchmark.sh \
    --configs examples/dynamo_integration/react_benchmark_agent/configs/multi_domain_25prompt_no_rethinking_test_train/*_eval.yml
```

Use `--concurrency N` to control parallelism (default: 4 per config,
5 configs = 20 concurrent streams across 8 workers).

### Outputs

Per-domain results:

```
outputs/dynamo_evals/multi_domain_test_train/{domain}_eval/
  jobs/job_{uuid}/
    standardized_data_all.csv        # Per-event profiler data
    workflow_output.json             # Agent responses
    tool_selection_quality_output.json  # TSQ scores
    prediction_trie.json             # Trie rebuilt from eval data
    inference_optimization.json      # Dynamo metrics (TTFT, ITL, KVE)
```

Comparison plots (auto-generated by the benchmark script):

```
outputs/dynamo_evals/multi_domain/comparison/
  summary_throughput_histograms.png
  throughput_histogram_per_llm_call_data.csv
  throughput_histogram_data.csv
```

### Verify

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/multi_domain/comparison/throughput_histogram_per_llm_call_data.csv')
print(f'Total LLM calls: {len(df)}')
print(f'Experiments: {df[\"experiment\"].nunique()}')
print(f'TTFT: median={df[\"ttft_ms\"].median():.1f}ms, p95={df[\"ttft_ms\"].quantile(0.95):.1f}ms')
print(f'TPS: median={df[\"tps\"].median():.1f} tok/s')
"
```

---

## Replay Logging (Optional)

For post-hoc analysis of routing decisions and learner evolution, set
`REPLAY_LOG_DIR` in `.env` before starting the Dynamo stack:

```bash
# In external/dynamo/.env
REPLAY_LOG_DIR=/workspace/logs/replay
```

This produces JSONL logs in `external/dynamo/logs/replay/`:

- `routing_decisions.jsonl` — per-decision: all 8 worker scores, feature
  vectors, chosen vs. native recommendation, override flag
- `routing_feedback.jsonl` — per-completion: TTFT, ITL, TPS, reward,
  beta/LinTS state after update
- `run_metadata.json` — run-level: router type, config, timestamp

These logs join on `decision_id` and enable:
- Accurate routing decision replay
- LinTS/Beta learning curve visualization
- Override rate (Thompson vs. native) over time
- KV overlap evolution per worker
- Feature vector analysis at each decision point

---

## Quick Reference

```bash
# Full pipeline (manual steps)
# 1. Generate tries
bash run_multi_domain_benchmark.sh --configs *_train.yml
bash collect_prediction_tries.sh --output-base <train_output_dir>

# 2. Optimize
nat optimize --config_file optimize_all_v0.yml

# 3. Accumulate learner state
curl -sf -X POST http://localhost:8084/state/reset
bash run_multi_domain_benchmark.sh --configs *_train.yml
curl -sf http://localhost:8084/state -o learner_state.json

# 4. Evaluate on held-out set
curl -sf -X POST http://localhost:8084/state/reset
curl -sf -X POST http://localhost:8084/state -H 'Content-Type: application/json' -d @learner_state.json
bash run_multi_domain_benchmark.sh --configs *_eval.yml --concurrency 8
```

## Known Limitations

- **Router config push fails** in read-only container filesystems. The
  optimizer applies `beta_decay`, `lints_v`, `lints_forget_rate` to live
  learner objects, but `ts_weight`, `temperature`, `cold_start_threshold`,
  `idle_boost` require writing `config.yaml` which fails with HTTP 500.
  These params only affect the NATS-RPC Thompson router, not the in-process
  `kv_thompson_native` scoring loop.

- **Prediction trie edge cases**: The last call in a conversation has
  `remaining_calls=0`, `interarrival_ms=0`, and potentially `output_tokens=0`.
  These are clamped to 1 in the transport code to pass downstream validators.

- **Scenario completion rate**: With 5 configs × max_concurrency scenarios
  competing for 8 workers, some scenarios may fail due to connection errors
  or recursion limits. Typical completion: 70-80%. Failed scenarios score 0
  in the evaluators, diluting the aggregate metrics.
