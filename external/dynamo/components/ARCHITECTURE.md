# Thompson Router Architecture

## Routing Modes

| Mode | Config Value | Description |
|------|-------------|-------------|
| `kv_native` | `router_type: kv_native` | Native KvRouter baseline (in-process, no learning) |
| `kv_thompson_native` | `router_type: kv_thompson_native` | Thompson + native KvRouter lifecycle (recommended) |
| `kv_thompson` | `router_type: kv_thompson` | Original Thompson via NATS RPC (legacy) |
| `kv_load` | `router_type: kv_load` | Dynamo formula replica (legacy, has cumulative counter bug) |

## Original Thompson Implementation (kv_thompson, via NATS RPC)

Two separate processes connected by NATS message queue. The processor
receives requests from the frontend, sends a routing RPC to the router,
then forwards to the chosen worker. Feedback flows back after completion.

```mermaid
flowchart LR
    subgraph frontend [Frontend]
        Client[HTTP Client]
    end

    subgraph proc [Processor Process]
        P_Recv["Receive\nPreprocessedRequest"]
        P_Hints["Extract hints\nprefix_id, osl, iat"]
        P_Route["NATS RPC\nfind_worker"]
        P_Direct["engine_client.direct\nto chosen worker"]
        P_FB["NATS RPC\nfeedback"]
    end

    subgraph router [Router Process]
        R_KvIdx["KvIndexer\nRadixTree + ZMQ events"]
        R_Metrics["Metrics Scraper\nHTTP poll per worker"]
        R_Score["Score Workers\noverlap * load_mod\n+ Beta sample\n+ LinTS sample\n+ affinity/switch"]
        R_Pending["PendingDecisions\n120s timeout sweep"]
        R_Learn["Update Learners\nBeta + LinTS"]
    end

    subgraph workers [vLLM Workers]
        W1[Worker 1]
        W2[Worker 2]
        W8[Worker 8]
    end

    Client -->|HTTP| P_Recv
    P_Recv --> P_Hints
    P_Hints -->|"NATS RPC\n~15ms P50"| P_Route
    P_Route -->|RouterRequest| R_KvIdx
    R_KvIdx --> R_Score
    R_Metrics --> R_Score
    R_Score -->|"worker_id\ndecision_id"| P_Route
    P_Route --> P_Direct
    P_Direct --> W1
    P_Direct --> W2
    P_Direct --> W8
    W1 -->|stream| P_FB
    P_FB -->|"NATS RPC\nlatency, tokens"| R_Pending
    R_Pending --> R_Learn
```

### Issues Found in Original Implementation
- `NameError` in router.py log statement crashed every response before yield (100% feedback timeout)
- Cumulative `_kv_load_routed` counter (never decrements) caused request funneling in kv_load mode
- Per-worker EMA baselines equalized rewards to ~0.5 (no worker differentiation)
- No Beta decay (learner couldn't adapt to changing conditions)
- Two NATS round-trips per request (~15ms P50 overhead)
- Separate metrics scraper HTTP polling (stale data)

## New Thompson Implementation (kv_thompson_native, in-process)

Single process with Dynamo's native KvRouter running in-process via pyo3
Python bindings. Thompson scoring uses native load signals and manages
lifecycle through KvRouter's `ActiveSequences`.

```mermaid
flowchart LR
    subgraph frontend [Frontend]
        Client[HTTP Client]
    end

    subgraph proc ["Processor Process (all-in-one)"]
        P_Recv["Receive\nPreprocessedRequest"]
        P_Hints["Extract hints\nprefix_id, osl, iat"]
        P_Loads["KvRouter.get_potential_loads\nprefill_tokens, decode_blocks\nper worker"]
        P_BW["KvRouter.best_worker\nnative recommendation\nquery-only"]
        P_Score["Thompson Scoring\nbase + Beta sample\n+ LinTS(features)\n+ session affinity"]
        P_Gen["KvRouter.generate\nworker_id=chosen\nauto lifecycle"]
        P_Learn["Update Learners\nBeta + LinTS\nglobal baseline reward"]
    end

    subgraph native ["Native KvRouter (Rust, in-process via pyo3)"]
        KV_Tree["Radix Tree\nKV overlap scoring"]
        KV_Active["ActiveSequences\nadd on route\nfree on completion"]
        KV_Fault["Fault Detection\nreport_instance_down"]
    end

    subgraph workers [vLLM Workers]
        W1[Worker 1]
        W2[Worker 2]
        W8[Worker 8]
    end

    Client -->|HTTP| P_Recv
    P_Recv --> P_Hints
    P_Hints -->|"in-process\n~3.5ms P50"| P_Loads
    P_Loads --> P_Score
    P_BW --> P_Score
    P_Hints --> P_BW
    P_Score --> P_Gen
    P_Gen -->|"route + stream\n+ mark_prefill\n+ free"| KV_Active
    KV_Active --> KV_Tree
    KV_Active --> KV_Fault
    P_Gen --> W1
    P_Gen --> W2
    P_Gen --> W8
    W1 -->|"stream complete\nobserve latency"| P_Learn
```

### Improvements Over Original
- All routing in-process (no NATS RPC, 3.5ms vs 15ms overhead)
- Native `ActiveSequences` lifecycle (add, mark_prefill_complete, free)
- `get_potential_loads()` replaces HTTP metrics scraper (accurate, in-flight aware)
- Global EMA baseline for reward (differentiates workers)
- Beta decay=0.995 (adapts to changing conditions, half-life ~138 observations)
- Session affinity via prefix_id tracking (4.2x TTFT improvement)
- No separate router process needed

## Learner Components

Extracted into `learners.py` for modularity and testability:

```mermaid
flowchart TD
    subgraph learners [learners.py]
        BL["BetaLearner\nPer-worker Beta-TS bandit\ndecay=0.995, window~200 obs"]
        LTS["LinTSLearner\n6-feature contextual bandit\nforget_rate=0.995"]
        LT["LatencyTracker\nGlobal EMA baseline\nreward = 1/(1+metric/baseline)"]
    end

    subgraph features ["Feature Vector (6-dim)"]
        F1["1.0 (bias)"]
        F2["inv_prefill: 1/(1+prefill_tokens/1000)"]
        F3["inv_decode: 1/(1+decode_blocks/50)"]
        F4["affinity: 1 if same worker, else 0"]
        F5["osl_norm: osl/1024"]
        F6["reuse_norm: tanh(0.25*reuse_budget)"]
    end

    subgraph feedback [Feedback Loop]
        Observe["Observe latency_ms\nand tokens_out"]
        Metric["metric = latency_ms / tokens_out"]
        Baseline["baseline = global EMA"]
        Reward["reward = 1/(1 + metric/baseline)"]
        Update["Update Beta + LinTS"]
    end

    features --> LTS
    LTS -->|"LinTS sample"| Score[Combined Score]
    BL -->|"Beta sample"| Score
    Observe --> Metric --> Reward
    Baseline --> Reward
    Reward --> Update
    Update --> BL
    Update --> LTS
    Observe --> LT
    LT --> Baseline
```

## File Structure

| File | Purpose |
|------|---------|
| `processor.py` | Request handler with routing modes (kv_native, kv_thompson_native, thompson) |
| `router.py` | Original Thompson router (NATS RPC server, used by kv_thompson mode) |
| `learners.py` | Modular learner components (BetaLearner, LinTSLearner, LatencyTracker, PendingDecisions) |
| `kv_indexer.py` | KV cache overlap scoring (RadixTree + ZMQ event listener) |
| `config.yaml` | All configurable parameters |
| `test_thompson_learners.py` | Unit tests for learner components (71 tests) |
| `test_convergence.py` | Convergence and adaptability tests (33 tests) |

## Configuration

Set `router_type` in `config.yaml` under `infrastructure:`:

```yaml
infrastructure:
  router_type: kv_thompson_native  # recommended
  block_size: 16
```

For the original NATS-based Thompson router, also start `router.py` and set:

```yaml
infrastructure:
  router_type: kv_thompson
```
