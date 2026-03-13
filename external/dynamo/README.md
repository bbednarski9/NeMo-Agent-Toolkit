<!--
Copyright (c) 2025-2026 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!-- path-check-skip-file -->

# Dynamo Backend Setup Guide

> [!NOTE]
> ⚠️ **EXPERIMENTAL**: This integration between NVIDIA NeMo Agent Toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice. We kindly ask that GitHub Issues are opened as bugs are issued quickly as features are subject to change.

> [!TIP]
> **Scope of This Guide**
>
> This document guides you through setting up and testing a NVIDIA NeMo Agent Toolkit-compatible Dynamo inference server on a Linux/CUDA machine. By the end of this guide, you will be able to make `curl` requests to the endpoint and receive inference outputs from the Dynamo server.
>
> For **end-to-end integration with NeMo Agent Toolkit workflows**, including detailed instructions and architectural considerations, see the [Dynamo Integration Examples](../../examples/dynamo_integration/README.md).

This guide covers setting up, running, and configuring the NVIDIA Dynamo backend for the React Benchmark Agent evaluations.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Starting Dynamo](#starting-dynamo)
4. [Building from Source](#building-from-source)
5. [Stopping Dynamo](#stopping-dynamo)
6. [Testing the Integration](#testing-the-integration)
7. [Monitoring](#monitoring)
8. [Dynamic Prefix Headers](#dynamic-prefix-headers)
9. [Configuration Reference](#configuration-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Dynamo is NVIDIA's high-performance LLM serving platform with KV cache optimization. The scope of the current integration is based around two core aspects. First, we have implemented a [Dynamo LLM](../../packages/nvidia_nat_core/src/nat/llm/dynamo_llm.py) support for NeMo Agent Toolkit inference on Dynamo runtimes. Second, we provide a set of startup scripts for Dynamo using VLLM and SgLang, supporting NeMo Agent Toolkit runtimes with the `nvext.agent_hints` and `nvext.cache_control` metadata for optimizing agentic performance on Dynamo servers. The following **Table** defines each script:

### Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMO BACKEND ARCHITECTURE                              │
└──────────────────────────────────────────────────────────────────────────────┘

                    NeMo Agent Toolkit Agent
                    (DynamoLLM / _type: dynamo)
                                │
                                │  POST /v1/chat/completions
                                │  Body includes:
                                │    nvext.agent_hints:
                                │      prefix_id, total_requests,
                                │      osl, iat, latency_sensitivity,
                                │      priority
                                │    nvext.cache_control:
                                │      type: "ephemeral", ttl
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DYNAMO FRONTEND                                     │
│                          Port 8099                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     HTTP API (OpenAI Compatible)                       │  │
│  │  ───────────────────────────────────────────────────────────────────── │  │
│  │  • /v1/chat/completions    - Chat completion endpoint                  │  │
│  │  • /v1/models              - List available models                     │  │
│  │  • /health                 - Health check                              │  │
│  │  • Rust nvext.rs parses body → AgentHints + CacheControl structs       │  │
│  │  • --enable-cache-control activates pin_prefix flow                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        ROUTING LAYER                                   │  │
│  │  ───────────────────────────────────────────────────────────────────── │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────────┐    │  │
│  │  │  Built-in KV Router      │  │  Thompson Sampling Router       │    │  │
│  │  │  (Rust/pyo3, default)    │  │  (components/, opt-in)          │    │  │
│  │  │  ────────────────────    │  │  ──────────────────────────     │    │  │
│  │  │  • Radix-tree KV match   │  │  • In-process via pyo3 KvRouter│    │  │
│  │  │  • Load-aware scoring    │  │  • LinTS + Beta bandits         │    │  │
│  │  │  • AgentHints.osl        │  │  • KV overlap + workload hints  │    │  │
│  │  │  • AgentHints.priority   │  │  • Session affinity tracking    │    │  │
│  │  │  • AgentHints.latency_   │  │  • Global EMA reward baseline   │    │  │
│  │  │    sensitivity           │  │  • Learns optimal routing       │    │  │
│  │  └──────────────────────────┘  └──────────────────────────────────┘    │  │
│  │                                                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐    │  │
│  │  │  Future: Workload-Aware Router                                 │    │  │
│  │  │  • Predictive scheduling using full agent_hints context        │    │  │
│  │  │  • Deeper osl / iat / latency_sensitivity integration          │    │  │
│  │  └────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     │  Route to selected worker
                                     │
        ┌────────────────────────────┴────────────────────────────────┐
        │                                                             │
        ▼                                                             ▼
┌─────────────────────────────┐                      ┌─────────────────────────────┐
│    vLLM WORKER              │         OR           │    SGLang WORKER            │
│    (Unified, TP=4)          │                      │    (Unified, TP=4)          │
│                             │                      │                             │
│  ┌───────────────────────┐  │                      │  ┌───────────────────────┐  │
│  │  vLLM Engine          │  │                      │  │  SGLang Engine        │  │
│  │  ─────────────────    │  │                      │  │  ─────────────────    │  │
│  │  • Model inference    │  │                      │  │  • Model inference    │  │
│  │  • KV Cache Management│  │                      │  │  • KV Cache Management│  │
│  │  • Token Generation   │  │                      │  │  • Token Generation   │  │
│  │  • Streaming Support  │  │                      │  │  • Streaming Support  │  │
│  │  • Priority scheduling│  │                      │  │  • Priority scheduling│  │
│  └───────────────────────┘  │                      │  └───────────────────────┘  │
│                             │                      │                             │
│  Cache Control:             │                      │  Cache Control:             │
│  • pin_prefix on generate   │                      │  • pin_prefix on generate   │
│  • TTL-based eviction       │                      │  • TTL-based eviction       │
└─────────────────────────────┘                      └─────────────────────────────┘
        │                                                             │
        └─────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  STREAMING RESPONSE  │
                           │  ────────────────────│
                           │  {"choices": [...],  │
                           │   "content": "..."}  │
                           └──────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE SERVICES                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐         ┌────────────────────────┐               │
│  │   ETCD                 │         │   NATS                 │               │
│  │   ──────────────────── │         │   ──────────────────── │               │
│  │   • Worker discovery   │         │   • Message queue      │               │
│  │   • Model card registry│         │   • Inter-component    │               │
│  │   • Health tracking    │         │     messaging          │               │
│  │   Port: 2379           │         │   • JetStream enabled  │               │
│  └────────────────────────┘         │   Port: 4222           │               │
│                                     └────────────────────────┘               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### NeMo Agent Toolkit Agent Hints

NeMo Agent Toolkit agents communicate workload characteristics to Dynamo through `nvext` fields in the OpenAI-compatible request body. These hints enable KV-aware routing, cache pinning, and priority scheduling. See [`components/NVEXT_FIELD_REFERENCE.md`](components/NVEXT_FIELD_REFERENCE.md) for the complete wire-format specification and config-to-wire mapping.

#### `nvext.agent_hints`

| Field | Description | Used in NAT Source/Components | Used in Dynamo Source/Components |
|---|---|---|---|
| `prefix_id` | Unique string identifying the KV cache prefix for a conversation | Yes — `DynamoPrefixContext` generates it; `processor.py` routes on it | No |
| `total_requests` | Expected number of LLM calls in this conversation | Yes — `processor.py` computes `reuse_budget` from it | No |
| `osl` | Expected output tokens (raw int or `LOW`/`MEDIUM`/`HIGH`) | Yes — `processor.py` uses it for router decode cost weighting. Validated for pass-through to Dynamo | Yes — native `AgentHints.osl` read by Dynamo's built-in frontend |
| `iat` | Expected inter-arrival time in milliseconds (raw int or `LOW`/`MEDIUM`/`HIGH`) | Yes — `processor.py` uses it for router stickiness weighting. Validated for pass-through to Dynamo | No |
| `latency_sensitivity` | How latency-sensitive this request is (from `@latency_sensitive` or prediction trie). Increase to prioritize a request. | Validated, passed through to Dynamo. Sets a context variable so we can decorate the context. | Yes — native `AgentHints.latency_sensitivity` → Dynamo queue ordering |
| `priority` | Engine scheduling priority (`max_sensitivity - latency_sensitivity`) | Validated, passed through to Dynamo | Yes — native `AgentHints.priority` → engine queue, eviction, preemption |

#### `nvext.cache_control`

| Field | Description | Used in NAT Source/Components | Used in Dynamo Source/Components |
|---|---|---|---|
| `type` | Cache pinning strategy. Only valid value is `"ephemeral"`. Required in JSON (no serde default on deserialization). | `dynamo_llm.py`: Injected client-side via `CachePinType.EPHEMERAL` (default). Configurable via `cache_pin_type` param; set to `None` to disable cache control entirely. | `nvext.rs`: Deserialized into `CacheControlType` enum (single variant: `Ephemeral`). Presence of `cache_control` triggers `pin_prefix` after generation in the KV push router. Requires `--enable-cache-control` on the frontend. |
| `ttl` | Duration string for how long to pin the prefix in the KV cache. Optional; defaults to `"5m"` (300s) when omitted. | `dynamo_llm.py`: Computed client-side as `total_requests * iat` (ms), converted to seconds, formatted as `"<N>m"` (whole minutes) or `"<N>s"`. The `iat` field is not consumed by Dynamo — it is only used here for this TTL computation and by the custom Thompson Sampling `processor.py`. | `nvext.rs` `CacheControl::ttl_seconds()`: Only parses `"5m"` (300s) and `"1h"` (3600s). Any other value logs a warning and falls back to 300s. The parsed TTL is forwarded as `ttl_seconds` to `pin_prefix` on the worker via the `cache_control` service mesh endpoint (`cache_control.rs` / `handler_base.py`). |

---

## Prerequisites

### Platform Requirements

> [!WARNING]
> **This example requires a Linux system with an NVIDIA GPU.** See the [Dynamo Support Matrix](https://docs.nvidia.com/dynamo/getting-started/support-matrix) for full details.
>
> **Supported Platforms:**
> - Ubuntu 22.04 / 24.04 (x86_64)
> - Ubuntu 24.04 (ARM64)
> - CentOS Stream 9 (x86_64, experimental)
>
> **Not Supported:**
> - ❌ macOS (Intel or Apple Silicon)
> - ❌ Windows
>
> You do **not** need to install `ai-dynamo` or `ai-dynamo-runtime` packages locally. The Dynamo server runs inside pre-built Docker images from NGC (`nvcr.io/nvidia/ai-dynamo/sglang-runtime`), which include all necessary components. By setting the environment variables `` and `` The NeMo Agent Toolkit Dynamo LLM client (`_type: dynamo`) is a pure HTTP client that works on any platform.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Architecture** | NVIDIA Hopper (H100) | B200 for higher throughput |
| **GPU Count** | 2 GPUs for small models (2 workers) | 8 GPUs for optimal performance |
| **GPU Memory** | 80GB per GPU (H100) | 192GB per GPU (B200) |
| **System RAM** | 256GB | 512GB+ |

### Software Requirements

> [!WARNING]
> This example requires a CUDA-compatible device with NVIDIA drivers installed. It cannot be run on systems without NVIDIA GPU hardware. You do not need to install ai-dynamo packages separately; the provided Docker images include them.

1. **Docker** installed and running (version 24.0+), with NVIDIA Container Toolkit
2. **NVIDIA Driver** with CUDA 12.0+ support, `nvidia-fabricmanager` enabled matching `NVIDIA-SMI` version. Verify with:

    ```bash
    docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 \
      bash -c "apt-get update && apt-get install -y python3-pip && pip3 install torch && python3 -c 'import torch; print(torch.cuda.is_available())'"
    ```

    The output should show `True`. If it shows `False` with error 802, ensure `nvidia-fabricmanager` is installed, running, and matches your driver version.

3. **Hugging Face CLI** for model downloads (optional, if model not already downloaded)
4. **Llama-3.3-70B-Instruct** model downloaded locally
5. **Python uv environment** python version 3.11-3.13


### uv Python Environment

```bash
cd /path/to/NeMo-Agent-Toolkit
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# install the NeMo Agent Toolkit
uv pip install -e ".[langchain]"
uv pip install -e examples/dynamo_integration/react_benchmark_agent
```

To activate an existing environment:

```bash
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
```

### Environment Variables

Before running the Dynamo scripts, configure the following environment variables. See `.env.example` for a complete list of all available options.

```bash
cd external/dynamo/

# Copy and customize the example environment file
cp .env.example .env

# Edit with your settings
vi .env

# Source the environment before running scripts
source .env
```

**OR** set variables directly:

```bash
export HF_HOME=/path/to/local/storage/.cache/huggingface

export HF_TOKEN=my_huggingface_read_token

# Required: Set your model directory path
export DYNAMO_MODEL_DIR=/path/to/your/models/Llama-3.3-70B-Instruct # or Llama-3.1-3B-Instruct for QA on H100 machines

# Optional: Set repository directory (for Thompson Sampling router)
export DYNAMO_REPO_DIR=/path/to/NeMo-Agent-Toolkit/external/dynamo

# Optional: Configure GPU devices (default: 0,1,2,3)
export DYNAMO_GPU_DEVICES=0,1,2,3
```

### Download model weights (can skip if already done)

```bash
[ -f .env ] && source .env || { echo "Warning: .env not found" >&2; false; }

# Change to the target model directory (create it if still needed)
cd "$(dirname "$DYNAMO_MODEL_DIR")"

# We will download the model weights directly from HuggingFace. See `NOTE` below.
uv pip install huggingface_hub
uv run huggingface-cli login  # Set or enter your HF token.
# OR: run it with python: `python -c "from huggingface_hub import login; login()"`

uv run huggingface-cli download "meta-llama/Llama-3.3-70B-Instruct" --local-dir "$DYNAMO_MODEL_DIR"
# OR: run it with python: `python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.3-70B-Instruct', local_dir='$DYNAMO_MODEL_DIR')"`
```

> [!NOTE]
> The Llama-3.3-70B-Instruct model requires approval from Meta. Request access at [huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) before downloading. You will need to create a HuggingFace Access Token with read access in order to download the model. On the `HuggingFace` website visit: "Access Tokens" -> "+ Create access token" to generate a token starting with `hf_`. Enter your token when prompted. Respond "n" when asked "Add token as git credential? (Y/n)". Set HF_HOME and HF_TOKEN in .env..

### Verify GPU Access

```bash
# Check NVIDIA driver and GPU availability
nvidia-smi

# Expected output should show:
# - At least 4 GPUs (H100 or B200)
# - CUDA version 12.0+
# - Sufficient free memory per GPU
```

Example output for an 8-GPU system:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:1B:00.0 Off |                    0 |
| N/A   31C    P0            187W / 1000W |  169082MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B200                    On  |   00000000:43:00.0 Off |                    0 |
| N/A   31C    P0            187W / 1000W |  169178MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B200                    On  |   00000000:52:00.0 Off |                    0 |
| N/A   36C    P0            193W / 1000W |  169230MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B200                    On  |   00000000:61:00.0 Off |                    0 |
| N/A   36C    P0            195W / 1000W |  169230MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B200                    On  |   00000000:9D:00.0 Off |                    0 |
| N/A   32C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B200                    On  |   00000000:C3:00.0 Off |                    0 |
| N/A   30C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B200                    On  |   00000000:D1:00.0 Off |                    0 |
| N/A   34C    P0            141W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B200                    On  |   00000000:DF:00.0 Off |                    0 |
| N/A   35C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

### Verify Docker and NVIDIA Container Toolkit

```bash
# Verify Docker is running
docker info
```

---

## Starting Dynamo

Startup scripts can be found in the same directory (`NeMo-Agent-Toolkit/external/dynamo/`) at this `README.md`

### Option 1: Default Rust Services (Development / Default)

Unified worker with Dynamo's built-in Rust frontend, processor, and router. Supports round-robin and KV-aware routing out of the box. Simpler setup, good for development and testing.

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# SGLang backend
bash start_dynamo_unified_sglang.sh > startup_output.txt 2>&1

# OR vLLM backend
bash start_dynamo_unified_vllm.sh > startup_output.txt 2>&1

# Wait for startup (watch GPU memory)
watch -n 1 nvidia-smi

# Verify Dynamo is running
curl -sv http://localhost:8099/health
# Expected: "HTTP/1.1 200 OK"

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Components started:**
- `etcd` container (`etcd-dynamo`) on port 2379
- `nats` container (`nats-dynamo`) on port 4222
- Dynamo container (`dynamo-sglang` or `dynamo-vllm`) with unified worker

**Routing modes available:** round-robin (default), KV-aware (enable with `ENABLE_KV_AWARE_ROUTING=true` and `DYNAMO_ENABLE_KV_EVENTS=true` in `.env`)

**Startup time**: Startup time may vary between 5-20 minutes for a 8-70B model, depending on the state of the system cache.

### Option 2: pyo3 KV-Aware + Thompson Sampling Router (Production / Research)

Unified worker with custom Python components that use pyo3 bindings to Dynamo's Rust KvRouter. Provides Thompson Sampling (LinTS + Beta bandits) for learning-based worker selection with KV overlap awareness, session affinity, and workload hints.

**Router type selection:** Set `router_type` in `components/config.yaml` under `infrastructure:`:
- `kv_native` — Dynamo's native KvRouter baseline (overlap + decode load scoring)
- `kv_thompson_native` — Thompson Sampling with KvRouter lifecycle (LinTS + Beta bandits, learning-based)

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# SGLang backend with Thompson router
bash start_dynamo_optimized_thompson_hints_sglang.sh > startup_output.txt 2>&1

# OR vLLM backend with Thompson router
bash start_dynamo_optimized_thompson_hints_vllm.sh > startup_output.txt 2>&1

# Wait for startup
watch -n 1 nvidia-smi

# Verify
curl -sv http://localhost:8099/health

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Additional features:**
- In-process KV-aware routing via pyo3 Python bindings to Rust KvRouter
- Thompson Sampling router (LinTS + Beta bandits) with learning-based worker selection
- KV cache overlap optimization with session affinity tracking
- Workload-aware routing based on `nvext.agent_hints` (osl, iat, latency_sensitivity)
- Global EMA reward baseline for worker differentiation

**KvThompsonRouter feature vector (9 dimensions):**

The LinTS contextual bandit uses a 9-dimensional feature vector per worker, derived from native KvRouter load signals and request hints:

| Index | Feature | Formula | Source |
|---|---|---|---|
| 0 | bias | `1.0` | constant |
| 1 | inv_load | `1 / (1 + decode_blocks / 50)` | KvRouter `get_potential_loads()` |
| 2 | overlap | `1 - prefill_tokens / max(1, tokens_in)` | KvRouter `get_potential_loads()` |
| 3 | affinity | `1.0` if same worker as last call, else `0.0` | prefix tracking |
| 4 | outstanding_norm | `tanh(0.1 * decode_blocks)` | KvRouter `get_potential_loads()` |
| 5 | decode_norm | `decode_cost(osl) / 3.0` | nvext hint (OSL interpolation: 128→1.0, 250→2.0, 1024→3.0) |
| 6 | prefill_norm | `tanh(prefill_tokens / 1024)` | KvRouter `get_potential_loads()` |
| 7 | iat_norm | `iat_factor(iat) / 1.5` | nvext hint (IAT interpolation: 50ms→1.5, 250ms→1.0, 1000ms→0.6) |
| 8 | reuse_norm | `tanh(0.25 * reuse_budget)` | nvext hint |

**Custom components location:** `components/`
- `config.yaml` - Thompson Sampling router configuration parameters
- `processor.py` - Request handler with routing modes, extracts `nvext` hints
- `router.py` - KvThompsonRouter class (in-process scoring, learner management HTTP server)
- `learners.py` - BetaLearner, LinTSLearner, LatencyTracker

---

## Building from Source

Instead of using pre-built NGC containers, you can build Dynamo runtime images directly from the [dynamo main branch](https://github.com/ai-dynamo/dynamo). This is useful for testing unreleased features or customizing the build.

The startup scripts (`start_dynamo_optimized_thompson_hints_vllm.sh` and `start_dynamo_optimized_thompson_hints_sglang.sh`) support source-built images through two `.env` variables:

- `DYNAMO_FROM_SOURCE=true` — enables source-build mode
- `DYNAMO_SOURCE_DIR` — path to local Dynamo source for patching (e.g., `/path/to/dynamo`)
- `DYNAMO_IMAGE` — the Docker image tag to build and use (for example, `dynamo-sglang-source:main`)

Set these in your `.env` file:

```bash
DYNAMO_FROM_SOURCE=true
DYNAMO_IMAGE="dynamo-sglang-source:main"   # or dynamo-vllm-source:main for vLLM
```

### Prerequisites for Building from Source

The build requires the following system packages on Ubuntu:

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config \
    libclang-dev protobuf-compiler python3-dev cmake
```

### Building the SGLang Runtime Image

Run the following commands from the root of the cloned dynamo repository:

```bash
cd /path/to/dynamo

# Render the SGLang Dockerfile from templates
python container/render.py --framework=sglang --target=runtime --output-short-filename

# Build the image (takes 30–90 minutes on first build; subsequent builds use cache)
docker build -t dynamo-sglang-source:main -f container/rendered.Dockerfile .
```

### Building the vLLM Runtime Image

```bash
cd /path/to/dynamo

# Render the vLLM Dockerfile from templates
python container/render.py --framework=vllm --target=runtime --output-short-filename

# Build the image
docker build -t dynamo-vllm-source:main -f container/rendered.Dockerfile .
```

### Running with the Source-Built Image

Once the image is built, run the startup script as normal — it automatically picks up `DYNAMO_FROM_SOURCE` and `DYNAMO_IMAGE` from `.env`:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# SGLang
bash start_dynamo_optimized_thompson_hints_sglang.sh > startup_output.txt 2>&1

# vLLM
bash start_dynamo_optimized_thompson_hints_vllm.sh > startup_output.txt 2>&1
```

If the image specified by `DYNAMO_IMAGE` does not exist, the script will print the exact build commands and exit with an error.

> **Note**: The dynamo `main` branch targets a different SGLang/vLLM version than the pre-built NGC containers. Verify the bundled framework version after building with `docker run --rm <image> python -c "import sglang; print(sglang.__version__)"`.

---

### Verifying the Integration

After starting Dynamo with any of the above options, verify the integration is working.

> [!NOTE]
> Commands in this section require the virtual environment to be active. See [uv Python Environment](#uv-python-environment).

#### Quick Validation with NeMo Agent Toolkit

Run simple workflows to test basic connectivity and prefix header support:

```bash
cd /path/to/NeMo-Agent-Toolkit

# Test basic Dynamo connectivity
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \
  --input "What time is it?"

# Test Dynamo with dynamic prefix headers (for Predictive KV-Aware Cache router)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml \
  --input "What time is it?"
```

#### Full Integration Test Suite

For comprehensive validation, run the integration test script:

> [!NOTE]
> Requires the virtual environment to be active. See [uv Python Environment](#uv-python-environment).

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash test_dynamo_integration.sh
```

**Environment variables** (optional):
- `DYNAMO_BACKEND` - Backend type: `sglang` # `vllm` and tensorRT still need to be developed
- `DYNAMO_MODEL` - Model name (default: `llama-3.3-70b`)
- `DYNAMO_PORT` - Frontend port (default: `8099`)

**Tests performed:**
1. NeMo Agent Toolkit environment is active
2. Configuration files exist
3. Dynamo frontend is responding on the configured port
4. Basic chat completion request works
5. Workflow with basic config runs successfully
6. Workflow with prefix hints runs successfully

**Expected output (all tests passing):**
```text
==========================================
Testing react_benchmark_agent with Dynamo
==========================================
Backend: sglang
Model: llama-3.3-70b
Port: 8099
==========================================

0. Checking if NAT environment is active...
✓ NAT command found

1. Checking if configuration files exist...
✓ Configuration files found

2. Checking if Dynamo frontend is running on port 8099...
✓ Dynamo frontend is running

3. Testing basic Dynamo endpoint...
✓ Dynamo endpoint is working

4. Testing NAT workflow with Dynamo (basic config)...
✓ Basic config test completed successfully

5. Testing NAT workflow with Dynamo (with prefix hints)...
✓ Prefix hints test completed successfully

==========================================
Test Summary
==========================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!
```

**What the test validates:**
1. The environment is activated
2. Configuration files exist
3. Dynamo frontend is running on port 8099
4. Dynamo endpoint responds correctly
5. Workflow executes with basic config
6. Workflow executes with prefix hints

If any tests fail, the script provides guidance on how to fix the issue.

---

## Stopping Dynamo

A single script stops all Dynamo components regardless of which mode was started:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash stop_dynamo.sh
```

**What it stops:**
- Dynamo container (`dynamo-sglang` or `dynamo-sglang-thompson`)
- `etcd` container (`etcd-dynamo`)
- `nats` container (`nats-dynamo`)

**Output:**
```text
=========================================================
Stopping Dynamo SGLang FULL STACK
=========================================================

Stopping Dynamo container (standard)...
✓ Dynamo container stopped and removed

Stopping ETCD container...
✓ ETCD container stopped and removed

Stopping NATS container...
✓ NATS container stopped and removed

=========================================================
✓ All components stopped!
=========================================================
```

---

## Testing the Integration

> [!NOTE]
> Commands in this section require the virtual environment to be active. See [uv Python Environment](#uv-python-environment).

#### Using NeMo Agent Toolkit (Recommended)

```bash
cd /path/to/NeMo-Agent-Toolkit

# Basic Dynamo test
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \
  --input "What time is it?"

# With prefix headers (for Thompson Sampling router)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml \
  --input "What time is it?"
```

#### Using curl

```bash
# Basic chat completion
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Streaming test
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

---

## Monitoring

### Interactive Monitor

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
./monitor_dynamo.sh
```

**Menu options:**
1. View Frontend logs
2. View Processor logs
3. View Router logs
4. View all component logs
5. View container logs
6. Test health endpoint
7. Test basic inference
8. Check GPU usage
9. Check process status

### Direct Commands

```bash
# View container logs
docker logs -f dynamo-sglang

# View `etcd` logs
docker logs -f etcd-dynamo

# View `nats` logs
docker logs -f nats-dynamo

# GPU utilization
watch -n 2 nvidia-smi

# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}"
```

---

## Dynamic Prefix Headers

When using the Thompson Sampling router (`start_dynamo_optimized_thompson_hints_sglang.sh` or `start_dynamo_optimized_thompson_hints_vllm.sh`), dynamic prefix headers enable optimal KV cache management and request routing.

### Overview

Prefix headers help the router:
- **Identify related requests** for KV cache reuse
- **Make routing decisions** based on workload characteristics
- **Track prefix state** for optimal worker selection
- **Improve throughput** through intelligent batching

### KV overlap routing: requirements and failure mode

Prefix headers do not include KV cache overlap. The router computes KV cache overlap scores by querying the backend through `dynamo.llm.KvIndexer`.

If overlap scores are unavailable, the router cannot account for KV cache match when routing and will behave like a non-KV-aware router for that signal.

This can happen in the following configuration:
- You are using a Dynamo image or build that does not include `dynamo.llm` KV routing classes. In this case, the router logs a warning that `dynamo.llm` is not available and overlap scores will be empty.

To confirm overlap scores are missing, check `router_metrics.csv` and verify that `overlap_chosen` is always `0.000000`.

### Configuration

Use the `dynamo` LLM type in your eval config. Prefix headers are sent by default:

```yaml
llms:
  dynamo_llm:
    _type: dynamo
    model_name: llama-3.3-70b
    base_url: http://localhost:8099/v1
    api_key: dummy

    # Prefix headers are enabled by default with template "nat-dynamo-{uuid}"
    # Optional: customize the template or routing hints
    # prefix_template: "react-benchmark-{uuid}"  # Custom template
    # prefix_template: null  # Set to null to disable prefix headers entirely
    prefix_total_requests: 10  # Expected requests per prefix
    prefix_osl: MEDIUM         # Output Sequence Length: LOW | MEDIUM | HIGH
    prefix_iat: MEDIUM         # Inter-Arrival Time: LOW | MEDIUM | HIGH
```

> **Note**: The `dynamo` LLM type automatically sends prefix headers using the default template `nat-dynamo-{uuid}`. To disable prefix headers entirely, set `prefix_template: null` in your config.

### Header Details

| Header | Description | Values |
|--------|-------------|--------|
| `x-prefix-id` | Unique identifier for request group | UUID-based string (null to disable all extra headers) |
| `x-prefix-total-requests` | Expected total requests for this prefix | Integer (1 for independent queries) |
| `x-prefix-osl` | Output Sequence Length hint | LOW (~50 tokens), MEDIUM (~200), HIGH (~500+) |
| `x-prefix-iat` | Inter-Arrival Time hint | LOW (rapid), MEDIUM (normal), HIGH (long delays) |

### Use Cases

#### Independent Queries (Evaluation)

Each question is independent, uses default prefix template:

```yaml
llms:
  eval_llm:
    _type: dynamo
    # prefix_template defaults to "nat-dynamo-{uuid}"
    prefix_total_requests: 1
    prefix_osl: MEDIUM
    prefix_iat: LOW  # Eval runs many queries quickly
```

#### Multi-Turn Conversations

Related requests should share a prefix:

```yaml
llms:
  chat_llm:
    _type: dynamo
    prefix_template: "chat-{uuid}"  # Optional: custom template
    prefix_total_requests: 8  # Average conversation length
    prefix_osl: MEDIUM
    prefix_iat: HIGH  # Users take time to type
```

#### Agent with Tool Calls

ReAct agents make multiple related calls:

```yaml
llms:
  agent_llm:
    _type: dynamo
    prefix_template: "agent-{uuid}"  # Optional: custom template
    prefix_total_requests: 5  # Typical tool call sequence
    prefix_osl: LOW   # Tool calls produce short responses
    prefix_iat: LOW   # Agent runs tool calls rapidly
```

### How It Works

1. **NeMo Agent Toolkit Configurations** uses `_type: dynamo` (prefix headers enabled by default)
2. **Dynamo LLM Provider** generates unique UUID per request using the template
3. **Headers injected** into HTTP request:
   ```text
   x-prefix-id: react-benchmark-a1b2c3d4e5f6g7h8
   x-prefix-total-requests: 1
   x-prefix-osl: MEDIUM
   x-prefix-iat: MEDIUM
   ```
4. **Dynamo Frontend** extracts headers
5. **Processor** tracks prefix state
6. **Router** makes routing decisions based on:
   - KV cache overlap with existing prefixes
   - Worker affinity for related requests
   - Load balancing across workers
   - Workload hints (OSL and IAT)

---

## Configuration Reference

### Environment Variables

The startup scripts support configuration through environment variables. Set these before running the scripts:

| Variable | Description | Default |
|----------|-------------|---------|
| `DYNAMO_MODEL_DIR` | Local path to the model directory | (required) |
| `DYNAMO_REPO_DIR` | Path to NeMo-Agent-Toolkit repository | Auto-detected |
| `DYNAMO_GPU_DEVICES` | Comma-separated GPU device IDs | `0,1,2,3` |
| `DYNAMO_HTTP_PORT` | Frontend HTTP port | `8099` |
| `DYNAMO_ETCD_PORT` | `etcd` client port | `2389` |
| `DYNAMO_NATS_PORT` | `nats` messaging port | `4232` |
| `DYNAMO_METRICS_URL` | Prometheus metrics endpoint URL for the router | `http://localhost:9090/metrics` |
| `ROUTER_METRICS_CSV` | Path to CSV file for router decision logging | `router_metrics.csv` |

Example configuration:

```bash
# Configure environment before running scripts
export DYNAMO_MODEL_DIR=/path/to/models/Llama-3.3-70B-Instruct
export DYNAMO_GPU_DEVICES=0,1,2,3
export DYNAMO_HTTP_PORT=8099

# Then start Dynamo (SGLang or vLLM)
bash start_dynamo_unified_sglang.sh
```

### Script Variables

Each startup script also has configurable variables at the top that can be edited directly:

```bash
# start_dynamo_unified_sglang.sh
CONTAINER_NAME="dynamo-sglang"
WORKER_GPUS="${DYNAMO_GPU_DEVICES:-0,1,2,3}"    # Override with env var or edit default
TP_SIZE=4
HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="llama-3.3-70b"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"
SHM_SIZE="16g"

# Infrastructure ports (non-default to avoid conflicts)
ETCD_CLIENT_PORT="${DYNAMO_ETCD_PORT:-2389}"
NATS_PORT="${DYNAMO_NATS_PORT:-4232}"

# Local paths - MUST be set via environment variable or edited here
LOCAL_MODEL_DIR="${DYNAMO_MODEL_DIR:?Error: DYNAMO_MODEL_DIR environment variable must be set}"
```

### Customizing GPU Assignment

Option 1: Use environment variable (recommended):

```bash
export DYNAMO_GPU_DEVICES=0,1,2,3
bash start_dynamo_unified_sglang.sh
```

Option 2: Edit the script directly:

```bash
# In the script, change:
WORKER_GPUS="0,1,2,3"

# The docker run command will use:
--gpus '"device=0,1,2,3"'
```

### Customizing Model

For a different model, update both the model directory and served name:

```bash
# Set environment variable for model path
export DYNAMO_MODEL_DIR="${HOME}/models/Llama-3.1-8B-Instruct"

# Edit script variables for model metadata
MODEL="/workspace/models/Llama-3.1-8B-Instruct"
SERVED_MODEL_NAME="llama-3.1-8b"
TP_SIZE=2  # Smaller models may need fewer GPUs
```

### Customizing Ports

Option 1: Use environment variables:

```bash
export DYNAMO_HTTP_PORT=8080
export DYNAMO_ETCD_PORT=2379
export DYNAMO_NATS_PORT=4222
bash start_dynamo_unified_sglang.sh
```

Option 2: Edit script directly:

```bash
HTTP_PORT=8080
ETCD_CLIENT_PORT=2379
NATS_PORT=4222
```

---

## Metrics CSV Files

The Thompson Sampling router (`start_dynamo_optimized_thompson_hints_sglang.sh` / `start_dynamo_optimized_thompson_hints_vllm.sh`) produces three CSV files for monitoring and analysis. These files are located in `/workspace/metrics/` inside the container.

### Accessing Metrics

```bash
# From the host
docker exec dynamo-sglang cat /workspace/metrics/router_metrics.csv
docker exec dynamo-sglang cat /workspace/metrics/processor_requests.csv
docker exec dynamo-sglang cat /workspace/metrics/frontend_throughput.csv

# From inside the container
docker exec -it dynamo-sglang bash
cat /workspace/metrics/router_metrics.csv
```

### router_metrics.csv

Logs every routing decision made by the Thompson Sampling router.

**Columns:**

| Column | Description |
|--------|-------------|
| `ts_epoch_ms` | Timestamp in milliseconds since epoch |
| `tokens_len` | Number of tokens in the request |
| `prefix_id` | Unique prefix identifier (auto-generated or from header) |
| `reuse_after` | Remaining reuse budget after this request |
| `chosen_worker` | Integer ID of the selected worker |
| `overlap_chosen` | KV cache overlap score (0.0-1.0) |
| `decode_cost` | Estimated `decode` cost |
| `prefill_cost` | Estimated `prefill` cost |
| `iat_level` | Inter-arrival time hint (LOW, MEDIUM, or HIGH) |
| `stickiness` | Worker affinity score |
| `load_mod` | Load modifier applied |

**Example output:**

```csv
ts_epoch_ms,tokens_len,prefix_id,reuse_after,chosen_worker,overlap_chosen,decode_cost,prefill_cost,iat_level,stickiness,load_mod
1767923263058,38,auto-9e05dbb0682f458a89b82f64bb328011,0,7587892060544177931,0.000000,2.000000,0.037109,MEDIUM,0.000,1.000000
```

### processor_requests.csv

Logs latency metrics for each processed request.

**Columns:**

| Column | Description |
|--------|-------------|
| `num_tokens` | Number of output tokens generated |
| `latency_ms` | Total request latency in milliseconds |
| `latency_ms_per_token` | Average latency per token |

**Example output:**

```csv
num_tokens,latency_ms,latency_ms_per_token
10,70152.021,7015.202100
```

### frontend_throughput.csv

Logs throughput metrics at regular intervals (default: every 5 seconds).

**Columns:**

| Column | Description |
|--------|-------------|
| `ts_epoch_ms` | Timestamp in milliseconds since epoch |
| `requests` | Number of requests completed in this interval |
| `interval_s` | Length of the measurement interval in seconds |
| `req_per_sec` | Computed requests per second |

**Example output:**

```csv
ts_epoch_ms,requests,interval_s,req_per_sec
1767923267849,0,5.000,0.000000
1767923272850,0,5.000,0.000000
1767923337856,1,5.000,0.200000
1767923342856,0,5.000,0.000000
```

---

## Troubleshooting

### Container Failed to Start

**Check logs:**
```bash
docker logs dynamo-sglang
```

**Common causes:**
- GPU not available
- Model path incorrect
- Port already in use

### Health Check Fails

```bash
# Check if container is running
docker ps --format '{{.Names}}'

# Check what's listening on port 8099
ss -tlnp | grep 8099
```

### `etcd` Connection Issues

```bash
# Check `etcd` health
curl http://localhost:2389/health

# Check `etcd` logs
docker logs etcd-dynamo
```

### `nats` Connection Issues

```bash
# Check `nats` is running
docker ps | grep nats-dynamo

# Check `nats` logs
docker logs nats-dynamo
```

### Workers Not Registering (0 / N Workers)

**Symptom**: Startup stalls at `Workers registered: 0 / N` even though container logs show workers initialized successfully (e.g., `Registered endpoint 'generate' with shared TCP server`).

**Cause**: `DYNAMO_WORKER_COMPONENT` mismatch. The startup script defaults to `backend`, but some Dynamo image versions register workers under the `worker` component name instead. The ETCD detection loop queries `v1/instances/workers/<component>/generate/` — if the component name doesn't match what workers actually register under, the count stays at 0.

**Diagnose**: Check what component name workers actually registered under:
```bash
curl -s http://localhost:2379/v3/kv/range -X POST \
  -H "Content-Type: application/json" \
  -d '{"key": "'$(echo -n "v1/instances/workers/" | base64 -w0)'", "range_end": "'$(echo -n "v1/instances/workers0" | base64 -w0)'", "keys_only": true}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); [print(base64.b64decode(kv['key']).decode()) for kv in d.get('kvs',[])]"
```

If keys show `workers/worker/generate/` instead of `workers/backend/generate/`, set in `.env`:
```bash
DYNAMO_WORKER_COMPONENT=worker
```

### Tokenizer Mismatch

**Symptom**: `KeyError: 'token_ids'` or tokenizer errors

**Fix**: Clear `etcd` data and restart
```bash
bash stop_dynamo.sh
# Wait a few seconds
bash start_dynamo_unified_sglang.sh  # or start_dynamo_unified_vllm.sh
```

### Slow Model Loading

**Symptom**: Takes 3+ minutes to start

**Causes:**
- 70B model takes ~90-120 seconds normally
- Cold cache may require model download
- Insufficient GPU memory causes swapping

**Monitoring:**
```bash
# Watch GPU memory during startup
watch -n 1 nvidia-smi
```

---

## File Structure

```text
external/dynamo/                                          # Dynamo backend
│
├── 📄 README.md                                          # This file - Dynamo setup guide
├── 📄 .env.example                                        # Example environment variables
├── 🔧 start_dynamo_unified_sglang.sh                    # SGLang + Rust services (round-robin / KV-aware)
├── 🔧 start_dynamo_unified_vllm.sh                      # vLLM + Rust services (round-robin / KV-aware)
├── 🔧 start_dynamo_optimized_thompson_hints_sglang.sh   # SGLang + pyo3 KV-aware + Thompson router
├── 🔧 start_dynamo_optimized_thompson_hints_vllm.sh     # vLLM + pyo3 KV-aware + Thompson router
├── 🔧 stop_dynamo.sh                                    # Stop all Dynamo services
├── 🔧 test_dynamo_integration.sh                        # Integration tests
├── 🔧 monitor_dynamo.sh                                 # Monitor running services
│
└── 📁 components/                                        # Custom router components
    ├── processor.py                                     # Request handler with routing modes
    ├── router.py                                        # Thompson Sampling router (NATS RPC)
    ├── learners.py                                      # BetaLearner, LinTSLearner, LatencyTracker
    ├── kv_indexer.py                                    # KV cache overlap scoring (RadixTree)
    ├── config.yaml                                      # Router configuration parameters
    └── NVEXT_FIELD_REFERENCE.md                         # nvext wire-format specification
```

---

## Quick Reference

### Commands

| Command | Description |
|---------|-------------|
| `bash start_dynamo_unified_sglang.sh` | SGLang + Rust services (round-robin / KV-aware) |
| `bash start_dynamo_unified_vllm.sh` | vLLM + Rust services (round-robin / KV-aware) |
| `bash start_dynamo_optimized_thompson_hints_sglang.sh` | SGLang + pyo3 KV-aware + Thompson router |
| `bash start_dynamo_optimized_thompson_hints_vllm.sh` | vLLM + pyo3 KV-aware + Thompson router |
| `bash stop_dynamo.sh` | Stop all services |
| `./test_dynamo_integration.sh` | Run integration tests |
| `./monitor_dynamo.sh` | Interactive monitoring |
| `curl localhost:8099/health` | Health check |
| `docker logs -f dynamo-sglang` | View logs |
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml --input "..."` | Quick NeMo Agent Toolkit validation |
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml --input "..."` | Test with prefix headers |

### Containers

| Container | Description |
|-----------|-------------|
| `dynamo-sglang` | SGLang Dynamo worker |
| `dynamo-vllm` | vLLM Dynamo worker |
| `etcd-dynamo` | Service discovery and model card registry |
| `nats-dynamo` | Inter-component messaging |

### Related Documentation

- **[React Benchmark Agent](../../examples/dynamo_integration/react_benchmark_agent/README.md)** - Complete evaluation guide
- **[Architecture](../../examples/dynamo_integration/ARCHITECTURE.md)** - System diagrams

---

## Next Steps

Now that you have a running Dynamo server and can make `curl` requests to the endpoint, you're ready to integrate with NeMo Agent Toolkit workflows.

> [!TIP]
> **Ready for Full Integration?**
>
> Visit the [Dynamo Integration Examples](../../examples/dynamo_integration/README.md) for:
> - End-to-end workflow integration with NeMo Agent Toolkit
> - Benchmark agent configurations and evaluation harnesses
> - Performance analysis scripts and visualization tools
> - Architectural deep-dives on toolkit-Dynamo integration patterns
