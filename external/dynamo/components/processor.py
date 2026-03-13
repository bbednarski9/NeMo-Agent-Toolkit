# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Optimized Processor for Thompson Sampling Router Architecture.

This processor uses the "Processor-as-Backend" pattern with DYNAMIC DISCOVERY
to intercept requests from the default Dynamo frontend and apply custom Thompson
Sampling routing.

## Dynamic Discovery Mode (Forward-Compatible)

Instead of using the deprecated `--static-endpoint` flag on the frontend, this
processor registers a model card in ETCD so the frontend can discover it via
its ModelWatcher. This is the forward-compatible approach.

### Requirements:
- Processor must be started with `--model-path` and `--model-name` arguments
- Model path must point to a valid model directory with tokenizer files
- Model name must match what the frontend expects (e.g., "llama-3.3-70b")

### Endpoint Registration Pattern

1. **This Processor registers as `dynamo.backend.generate`** - Dynamically with instance ID
2. **Processor calls `register_llm()`** - Advertises model card in ETCD
3. **Frontend's ModelWatcher discovers us** - Routes requests to our endpoint
4. **SGLang Worker registers as `workers.worker.generate`** - We forward to actual workers

## Request Flow

```
Frontend (discovers backends via ETCD ModelWatcher)
    → routes to dynamo.backend.generate-{instance_id}
    → THIS PROCESSOR (discovered via model card!)
        → extracts hints from nvext annotations
        → routes via in-process KvRouter (kv_native or kv_thompson) → worker_id
        → forwards to workers.worker.generate (actual SGLang workers)
```

Key differences from generalized/processor.py:
- Uses dynamic discovery (no --static-endpoint on frontend)
- Registers model card via register_llm() for ETCD discovery
- Registers as `dynamo.backend.generate` (not `dynamo.processor.process`)
- Forwards to `workers.worker.generate` (workers in separate namespace)
- Receives PreprocessedRequest instead of ChatCompletionRequest
- Extracts hints from nvext annotations (prefix_id:value format)
- Uses Dynamo metrics API for Prometheus integration (auto-exposed at /metrics)
- No tokenization (handled by frontend preprocessor)

## Metrics

All metrics are exposed via Dynamo's `/metrics` endpoint (requires DYN_SYSTEM_PORT).
Metrics use the `dynamo_component_` prefix and include standard Dynamo labels:
- `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`

Custom metrics for Thompson Sampling routing:
- `requests_total` - Total requests processed
- `request_latency_seconds` - End-to-end request latency histogram
- `tokens_in_total` / `tokens_out_total` - Token throughput counters
- `routing_decisions_total` - Per-worker routing decision counter
- `router_errors_total` / `engine_errors_total` - Error counters
- `active_requests` - Current in-flight request gauge

KV Cache Efficiency (KVE) metrics:
- `kve_prompt_tokens_total` - Total prompt tokens (efficiency denominator)
- `kve_cached_tokens_total` - Total cached tokens hit (efficiency numerator)
- `kve_device_blocks_total` - Cache hits from device (GPU) memory
- `kve_host_blocks_total` - Cache hits from host (CPU) memory
- `kve_disk_blocks_total` - Cache hits from disk

## Grafana Integration

Metrics are exposed at `/metrics` in Prometheus format. Enable with:
  DYN_SYSTEM_PORT=8081 python processor.py --model-path ... --model-name ...

Full metric names include the `dynamo_component_` prefix:
  dynamo_component_requests_total{dynamo_namespace="dynamo",dynamo_component="backend",dynamo_endpoint="generate"}

Example PromQL queries for Grafana dashboards:
  # KV Cache Efficiency (%)
  rate(dynamo_component_kve_cached_tokens_total[5m]) / rate(dynamo_component_kve_prompt_tokens_total[5m]) * 100

  # Request latency p99
  histogram_quantile(0.99, rate(dynamo_component_request_latency_seconds_bucket[5m]))

## Data Source Requirements

KVE metrics require the underlying engine to return cache efficiency data:
- `usage.prompt_tokens_details.cached_tokens` - Standard OpenAI field (should work with prefix caching enabled)
- `nvext.cache_hit_breakdown` - Engine-specific extension (NOT standard Dynamo NvExt)
"""

import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import Any

import uvloop
from dynamo.llm import ModelInput
from dynamo.llm import ModelType
from dynamo.llm import register_llm
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging


def _get_endpoint(runtime: DistributedRuntime, namespace: str, component: str, endpoint: str):
    """Get a Dynamo endpoint, compatible with both old and new runtime APIs.

    Old API (NGC images): runtime.namespace("ns").component("comp").endpoint("ep")
    New API (source build): runtime.endpoint("ns.comp.ep")
    """
    if hasattr(runtime, "namespace"):
        return runtime.namespace(namespace).component(component).endpoint(endpoint)
    return runtime.endpoint(f"{namespace}.{component}.{endpoint}")
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest
configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ----------------------- KV efficiency data ----------------------- #
class KVEfficiencyData:
    """
    Container for KV cache efficiency data extracted from worker responses.

    This data is used to compute and publish KVE metrics asynchronously,
    ensuring zero impact on routing throughput.
    """

    __slots__ = ("prompt_tokens", "cached_tokens", "device_blocks", "host_blocks", "disk_blocks")

    def __init__(self):
        self.prompt_tokens: int = 0
        self.cached_tokens: int = 0
        self.device_blocks: int = 0
        self.host_blocks: int = 0
        self.disk_blocks: int = 0

    def has_data(self) -> bool:
        """Check if any KVE data was collected."""
        return self.prompt_tokens > 0

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "KVEfficiencyData":
        """
        Extract KVE data from a worker response chunk.

        Expected fields in response (OpenAI-compatible):
        - usage.prompt_tokens: Total prompt tokens
        - usage.prompt_tokens_details.cached_tokens: Cached token count

        Optional engine-specific fields (may not be present):
        - nvext.cache_hit_breakdown.{device,host,disk}_blocks: Per-tier hits

        Note: cache_hit_breakdown is NOT a standard Dynamo NvExt field.
        It must be enabled/configured in the underlying engine (vLLM/SGLang).
        """
        kve = cls()

        # Extract from usage field (OpenAI-compatible, should always work)
        usage = data.get("usage")
        if isinstance(usage, dict):
            kve.prompt_tokens = usage.get("prompt_tokens", 0) or 0
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                kve.cached_tokens = prompt_details.get("cached_tokens", 0) or 0

        # Extract cache breakdown from nvext (engine-specific, may not be present)
        # This is NOT a standard Dynamo NvExt field - requires engine configuration
        nvext = data.get("nvext")
        if isinstance(nvext, dict):
            breakdown = nvext.get("cache_hit_breakdown")
            if isinstance(breakdown, dict):
                kve.device_blocks = breakdown.get("device_blocks", 0) or 0
                kve.host_blocks = breakdown.get("host_blocks", 0) or 0
                kve.disk_blocks = breakdown.get("disk_blocks", 0) or 0

        return kve


# ----------------------- metrics dataclass ----------------------- #
class ProcessorMetrics:
    """
    Container for Thompson Sampling processor metrics.

    Metrics are created via prometheus_client and exposed on Dynamo's /metrics
    endpoint through RuntimeMetrics.register_prometheus_expfmt_callback().

    In Dynamo 0.9.0 the old endpoint.metrics.create_intcounter() API was removed.
    We use a private CollectorRegistry to avoid collisions with other components
    and register a callback that returns exposition text for each scrape.
    """

    def __init__(self, endpoint):
        """
        Initialize metrics using prometheus_client.

        Args:
            endpoint: Dynamo endpoint object providing the metrics interface.
        """
        # Private registry so we don't collide with vLLM or Dynamo metrics
        self._registry = CollectorRegistry()
        prefix = "dynamo_component_thompson"

        # Request throughput
        self.requests_total = Counter(
            f"{prefix}_requests_total",
            "Total requests processed by the Thompson Sampling processor",
            registry=self._registry,
        )

        # Latency histogram
        self.request_latency_seconds = Histogram(
            f"{prefix}_request_latency_seconds",
            "End-to-end request latency in seconds",
            registry=self._registry,
        )

        # Token throughput
        self.tokens_in_total = Counter(
            f"{prefix}_tokens_in_total",
            "Total input tokens processed",
            registry=self._registry,
        )
        self.tokens_out_total = Counter(
            f"{prefix}_tokens_out_total",
            "Total output tokens generated",
            registry=self._registry,
        )

        # Routing decisions by worker (for analyzing load distribution)
        self.routing_decisions_total = Counter(
            f"{prefix}_routing_decisions_total",
            "Routing decisions by worker",
            ["worker_id"],
            registry=self._registry,
        )

        # Error tracking
        self.router_errors_total = Counter(
            f"{prefix}_router_errors_total",
            "Router communication errors (failed to pick worker)",
            registry=self._registry,
        )
        self.engine_errors_total = Counter(
            f"{prefix}_engine_errors_total",
            "Backend engine errors (failed during streaming)",
            registry=self._registry,
        )

        # Active request gauge
        self.active_requests = Gauge(
            f"{prefix}_active_requests",
            "Currently active requests being processed",
            registry=self._registry,
        )

        # -----------------------------------------------------------------
        # KV Cache Efficiency (KVE) metrics
        # These track cache hit rates for analyzing routing effectiveness.
        # Efficiency = kve_cached_tokens_total / kve_prompt_tokens_total
        # -----------------------------------------------------------------
        self.kve_prompt_tokens_total = Counter(
            f"{prefix}_kve_prompt_tokens_total",
            "Total prompt tokens processed (KV efficiency denominator)",
            registry=self._registry,
        )
        self.kve_cached_tokens_total = Counter(
            f"{prefix}_kve_cached_tokens_total",
            "Total cached tokens hit (KV efficiency numerator)",
            registry=self._registry,
        )

        # Cache hit breakdown by memory tier (for analyzing cache hierarchy)
        self.kve_device_blocks_total = Counter(
            f"{prefix}_kve_device_blocks_total",
            "KV cache blocks hit from device (GPU) memory",
            registry=self._registry,
        )
        self.kve_host_blocks_total = Counter(
            f"{prefix}_kve_host_blocks_total",
            "KV cache blocks hit from host (CPU) memory",
            registry=self._registry,
        )
        self.kve_disk_blocks_total = Counter(
            f"{prefix}_kve_disk_blocks_total",
            "KV cache blocks hit from disk storage",
            registry=self._registry,
        )

        # Register the callback so Dynamo exposes these at /metrics
        endpoint.metrics.register_prometheus_expfmt_callback(self._generate_metrics)

        logger.info("Processor metrics initialized via prometheus_client + RuntimeMetrics callback")

    def _generate_metrics(self) -> str:
        """Return Prometheus exposition text for all Thompson metrics."""
        return generate_latest(self._registry).decode("utf-8")


# -------------------------- processor handler -------------------------- #
class ProcessorRequestHandler:
    """
    Processor that receives PreprocessedRequest from the default Dynamo frontend,
    extracts routing hints from nvext annotations, and coordinates with the
    native KvRouter for worker selection.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        endpoint,
        enable_router: bool = True,
        routing_mode: str = "kv_thompson",
        model_name: str = "",
        kv_block_size: int = 16,
    ):
        """
        Initialize the processor request handler.

        Args:
            runtime: Dynamo distributed runtime for client connections.
            endpoint: Dynamo endpoint for metrics registration.
            enable_router: Whether to use any router (default: True).
            routing_mode: "kv_native" (Dynamo's native KvRouter in-process) or
                          "kv_thompson" (Thompson learners + native KvRouter lifecycle).
            model_name: Served model name (needed for KvRouter.generate()).
            kv_block_size: KV cache block size (needed for KvRouter init).
        """
        self.runtime = runtime
        self.endpoint = endpoint
        self.enable_router = enable_router
        self.routing_mode = routing_mode
        self.model_name = model_name
        self.kv_block_size = kv_block_size

        # Client connections (initialized in initialize())
        self.engine_client = None
        self.kv_router = None  # Native KvRouter (pyo3)
        self.thompson_router = None  # KvThompsonRouter (kv_thompson mode only)

        # Prefix-level state: {prefix_id: {"total": int, "processed": int}}
        self._prefix_state: dict[str, dict[str, int]] = {}
        self._prefix_lock = asyncio.Lock()

        # Prevent fire-and-forget tasks from being garbage-collected
        self._background_tasks: set[asyncio.Task] = set()

        # Metrics (initialized in initialize())
        self._metrics: ProcessorMetrics | None = None

        # Replay logger for post-hoc routing analysis (initialized in initialize())
        self._replay_logger = None  # type: ReplayLogger | None

    async def initialize(self):
        """Initialize processor by setting up metrics and connecting to services."""
        # Initialize metrics using Dynamo's metrics API
        self._metrics = ProcessorMetrics(self.endpoint)

        # Connect to actual workers at workers.{component}.generate
        worker_component_name = os.environ.get("DYNAMO_WORKER_COMPONENT")
        if not worker_component_name:
            raise ValueError("DYNAMO_WORKER_COMPONENT environment variable is required. "
                             "Set to 'worker' for SGLang or 'backend' for vLLM.")
        worker_endpoint = _get_endpoint(self.runtime, "workers", worker_component_name, "generate")
        self.engine_client = await worker_endpoint.client()
        logger.info("Engine client created for workers/%s/generate, waiting for worker instances...",
                    worker_component_name)
        await self.engine_client.wait_for_instances()
        logger.info("Workers discovered: %s", list(self.engine_client.instance_ids()))

        if self.enable_router and self.routing_mode in ("kv_native", "kv_thompson"):
            try:
                from dynamo.llm import KvRouter, KvRouterConfig
                kv_config = KvRouterConfig()
                self.kv_router = KvRouter(
                    endpoint=worker_endpoint,
                    block_size=self.kv_block_size,
                    kv_router_config=kv_config,
                )
                logger.info(
                    "Native KvRouter initialized (block_size=%d, mode=%s)",
                    self.kv_block_size, self.routing_mode,
                )
            except ImportError:
                raise ImportError(
                    "KvRouter not available — requires source-built Dynamo image. "
                    "Cannot proceed with routing_mode=%s." % self.routing_mode
                )

        if self.enable_router and self.routing_mode == "kv_thompson":
            from router import KvThompsonRouter
            config_path = os.environ.get("ROUTER_CONFIG_PATH", "/workspace/custom_dynamo/config.yaml")
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    router_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Could not read config at %s (%s), using defaults", config_path, e)
                router_config = {}
            self.thompson_router = KvThompsonRouter(self.kv_router, config=router_config)
            logger.info("KvThompsonRouter initialized (mode=kv_thompson)")

        logger.info("Processor initialized (routing_mode=%s, workers=%s/generate)",
                    self.routing_mode, worker_component_name)

        hint_overrides = {
            k: os.environ[k]
            for k in ("HINT_OVERRIDE_OSL", "HINT_OVERRIDE_IAT",
                       "HINT_OVERRIDE_TOTAL_REQUESTS", "HINT_OVERRIDE_PREFIX_ID")
            if k in os.environ
        }
        if hint_overrides:
            logger.warning("Hint overrides ACTIVE — extracted agent_hints will be clamped: %s", hint_overrides)
        else:
            logger.info("No HINT_OVERRIDE_* env vars set; agent_hints pass through unchanged")

        replay_dir = os.environ.get("REPLAY_LOG_DIR")
        if replay_dir:
            from datetime import datetime, timezone

            from replay_logger import ReplayLogger
            run_id = f"{self.routing_mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            self._replay_logger = ReplayLogger(
                output_dir=replay_dir,
                run_id=run_id,
                router_type=self.routing_mode,
                config={
                    "block_size": self.kv_block_size,
                    "num_workers": len(list(self.engine_client.instance_ids())),
                },
            )

    # ---- annotation extraction ----
    @staticmethod
    def _extract_annotation(annotations: list[str], key: str, default: str | None = None) -> str | None:
        """Extract value from annotations list (format: 'key:value')."""
        prefix = f"{key}:"
        for ann in annotations:
            if ann.startswith(prefix):
                return ann[len(prefix):]
        return default

    @staticmethod
    def _to_category(
        value: str | None,
        thresholds: tuple[float, float],
        default: str = "MEDIUM",
    ) -> str:
        """Convert a value to LOW/MEDIUM/HIGH category.

        Accepts either a categorical string (LOW/MEDIUM/HIGH) directly, or a
        numeric string which is converted using the given thresholds::

            value < thresholds[0]  → LOW
            value < thresholds[1]  → MEDIUM
            value >= thresholds[1] → HIGH

        Values are always raw integers.
        """
        if not value:
            return default
        upper = value.strip().upper()
        if upper in ("LOW", "MEDIUM", "HIGH"):
            return upper
        # Try numeric conversion
        try:
            num = float(value)
            if num < thresholds[0]:
                return "LOW"
            if num < thresholds[1]:
                return "MEDIUM"
            return "HIGH"
        except (ValueError, TypeError):
            return default

    def _extract_hints(self, request: dict[str, Any]) -> tuple[str, int, int, int]:
        """
        Extract routing hints from PreprocessedRequest.

        Supports two formats:
          - Dynamo main: ``routing.expected_output_tokens``, ``routing.priority_jump``
          - Legacy annotations: ``"osl:250"``, ``"iat:MEDIUM"``, ``"prefix_id:..."``

        ``routing`` fields take precedence when present; annotations are the
        fallback for Thompson-specific hints (prefix_id, total_requests) that
        have no routing equivalent.

        Returns: (prefix_id, total_requests, osl, iat)
        """
        annotations = request.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []
        routing = request.get("routing") or {}
        if not isinstance(routing, dict):
            routing = {}

        if annotations:
            logger.debug("Raw annotations: %s", annotations)
        if routing:
            logger.debug("Routing hints: %s", routing)

        # --- prefix_id (annotations only, no routing equivalent) ---
        prefix_id = self._extract_annotation(annotations, "prefix_id")
        if not prefix_id:
            prefix_id = f"auto-{uuid.uuid4().hex}"
            logger.debug("No prefix_id in annotations, generated: %s", prefix_id)

        # --- total_requests (annotations only) ---
        total_str = self._extract_annotation(annotations, "total_requests", "1")
        try:
            total_requests = max(1, int(total_str))
        except (ValueError, TypeError):
            total_requests = 1

        # --- osl: routing.expected_output_tokens (u32) → annotations fallback ---
        _OSL_CAT = {"LOW": 128, "MEDIUM": 250, "HIGH": 1024}
        routing_osl = routing.get("expected_output_tokens")
        if routing_osl is not None:
            try:
                osl: int = max(1, int(routing_osl))
            except (ValueError, TypeError):
                osl = 250
        else:
            osl_raw = self._extract_annotation(annotations, "osl", "MEDIUM")
            try:
                osl = int(osl_raw)
            except (ValueError, TypeError):
                osl = _OSL_CAT.get(str(osl_raw).upper(), 250)

        # --- iat: routing.priority_jump (seconds) → convert to ms; annotations fallback ---
        _IAT_CAT = {"LOW": 50, "MEDIUM": 250, "HIGH": 1000}
        routing_iat = routing.get("priority_jump")
        if routing_iat is not None:
            try:
                iat: int = max(1, int(float(routing_iat) * 1000.0))
            except (ValueError, TypeError):
                iat = 250
        else:
            iat_raw = self._extract_annotation(annotations, "iat", "MEDIUM")
            try:
                iat = int(iat_raw)
            except (ValueError, TypeError):
                iat = _IAT_CAT.get(str(iat_raw).upper(), 250)

        # Apply env-var hint overrides for A/B testing.
        # When set, these clamp extracted hints to fixed neutral values so
        # the Thompson router receives no useful signal from that dimension.
        env_osl = os.environ.get("HINT_OVERRIDE_OSL")
        if env_osl is not None:
            osl = int(env_osl)
        env_iat = os.environ.get("HINT_OVERRIDE_IAT")
        if env_iat is not None:
            iat = int(env_iat)
        env_total = os.environ.get("HINT_OVERRIDE_TOTAL_REQUESTS")
        if env_total is not None:
            total_requests = max(1, int(env_total))
        env_prefix = os.environ.get("HINT_OVERRIDE_PREFIX_ID")
        if env_prefix and env_prefix.lower() == "auto":
            prefix_id = f"auto-{uuid.uuid4().hex}"

        return prefix_id, total_requests, osl, iat

    async def _update_prefix_state(self, prefix_id: str, total_requests: int) -> int:
        """
        Update prefix counters and return remaining_after (reuse_budget).

        This tracks how many requests remain for a given prefix, allowing the
        router to make informed decisions about KV cache placement.
        """
        async with self._prefix_lock:
            state = self._prefix_state.get(prefix_id)
            if state is None:
                state = {"total": total_requests, "processed": 0}
                self._prefix_state[prefix_id] = state
            else:
                # Update total if a higher count is reported
                state["total"] = max(state["total"], total_requests)

            state["processed"] += 1
            remaining_after = max(state["total"] - state["processed"], 0)

            # Clean up completed prefixes immediately
            if remaining_after == 0:
                self._prefix_state.pop(prefix_id, None)

        return remaining_after

    def _update_kve_metrics_sync(self, kve: KVEfficiencyData) -> None:
        """
        Update KV cache efficiency metrics (synchronous, called from background task).

        This is intentionally synchronous - counter increments are atomic and
        extremely fast (microseconds). The async wrapper exists only to allow
        fire-and-forget scheduling via create_task().
        """
        if not kve.has_data():
            return

        # Update counters - these are atomic operations
        self._metrics.kve_prompt_tokens_total.inc(kve.prompt_tokens)
        self._metrics.kve_cached_tokens_total.inc(kve.cached_tokens)
        self._metrics.kve_device_blocks_total.inc(kve.device_blocks)
        self._metrics.kve_host_blocks_total.inc(kve.host_blocks)
        self._metrics.kve_disk_blocks_total.inc(kve.disk_blocks)

        # Log efficiency for debugging (only if we have meaningful data)
        if kve.prompt_tokens > 0:
            efficiency = kve.cached_tokens / kve.prompt_tokens * 100
            logger.debug(
                "KVE update: prompt=%d cached=%d eff=%.1f%% (dev=%d host=%d disk=%d)",
                kve.prompt_tokens,
                kve.cached_tokens,
                efficiency,
                kve.device_blocks,
                kve.host_blocks,
                kve.disk_blocks,
            )

    async def _update_kve_metrics_async(self, kve: KVEfficiencyData) -> None:
        """
        Async wrapper for KVE metric updates (fire-and-forget via create_task).

        This allows the main streaming path to continue without waiting for
        metric updates, ensuring zero impact on routing throughput.
        """
        try:
            self._update_kve_metrics_sync(kve)
        except Exception:
            # Never let metric updates crash the system
            logger.exception("Failed to update KVE metrics")

    # ---- main generation endpoint ----
    async def generate(self, raw: dict[str, Any]):
        """
        Processor endpoint: receives PreprocessedRequest from frontend.

        Expected format (from Dynamo preprocessor):
        {
            "token_ids": [...],
            "annotations": ["prefix_id:xyz", "total_requests:10", ...],
            "sampling_options": {...},
            "stop_conditions": {...},
            ...
        }
        """
        # Track active requests
        self._metrics.active_requests.inc()

        try:
            # Increment request counter
            self._metrics.requests_total.inc()
            t_proc_start = time.perf_counter()

            # Extract routing hints from annotations
            prefix_id, total_requests, osl, iat = self._extract_hints(raw)

            # Get token IDs from preprocessed request
            token_ids = raw.get("token_ids", [])
            if not isinstance(token_ids, list):
                token_ids = []

            tokens_in = len(token_ids)
            is_auto = prefix_id.startswith("auto-")

            # Compute reuse_budget := remaining AFTER this request
            reuse_budget = await self._update_prefix_state(prefix_id, total_requests)

            logger.info(
                "Processing request: prefix=%s total=%d reuse_budget=%d osl=%s iat=%s "
                "tokens=%d source=%s",
                prefix_id,
                total_requests,
                reuse_budget,
                osl,
                iat,
                tokens_in,
                "auto" if is_auto else "annotation",
            )

            if self.routing_mode == "kv_thompson" and self.thompson_router is not None:
                # ---- Thompson + Native KvRouter lifecycle ----
                t_routing_start = time.perf_counter()

                decision = await self.thompson_router.pick_worker(
                    token_ids, prefix_id, reuse_budget, osl, iat, tokens_in,
                )
                chosen = decision.chosen

                decision_id = None
                if self._replay_logger is not None:
                    decision_id = str(uuid.uuid4())
                    session_depth = total_requests - reuse_budget
                    chosen_x = decision.features
                    self._replay_logger.log_decision({
                        "event": "route",
                        "decision_id": decision_id,
                        "run_id": self._replay_logger.run_id,
                        "session_id": prefix_id,
                        "llm_call_idx": session_depth - 1,
                        "session_depth": session_depth,
                        "timestamp_ns": time.time_ns(),
                        "chosen_worker": chosen,
                        "native_recommendation": decision.native_pick,
                        "overrode_native": chosen != decision.native_pick,
                        "features": {
                            "inv_prefill": round(float(chosen_x[1]), 4),
                            "inv_decode": round(float(chosen_x[2]), 4),
                            "affinity": float(chosen_x[3]),
                            "osl_norm": round(float(chosen_x[4]), 4),
                            "reuse_norm": round(float(chosen_x[5]), 4),
                            "iat_norm": round(float(chosen_x[6]), 4),
                        } if chosen_x is not None else None,
                        "workers": decision.worker_details,
                    })

                routing_ms = (time.perf_counter() - t_routing_start) * 1000.0

                logger.info(
                    "KvThompson routing: prefix=%s chosen=%s native_pick=%s "
                    "workers=%d routing_ms=%.1f",
                    prefix_id, chosen, decision.native_pick,
                    len(decision.worker_details), routing_ms,
                )

                stop_conditions = raw.get("stop_conditions")
                sampling_options = raw.get("sampling_options")

                stream = await self.kv_router.generate(
                    token_ids=token_ids,
                    model=self.model_name,
                    stop_conditions=stop_conditions,
                    sampling_options=sampling_options,
                    worker_id=chosen,
                )

                t0 = time.perf_counter()
                t_first_token = None
                tokens_out = 0
                async for chunk in stream:
                    if isinstance(chunk, dict):
                        data = chunk
                    else:
                        data = chunk.data() if hasattr(chunk, "data") else chunk

                    if "token_ids" in data and isinstance(data["token_ids"], list):
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        tokens_out += len(data["token_ids"])

                    yield data

                    if "finish_reason" in data and data["finish_reason"] is not None:
                        latency_seconds = time.perf_counter() - t0
                        latency_ms = latency_seconds * 1000.0

                        fb = self.thompson_router.update_feedback(decision, latency_ms, tokens_out)

                        self._metrics.request_latency_seconds.observe(latency_seconds)
                        self._metrics.tokens_in_total.inc(tokens_in)
                        self._metrics.tokens_out_total.inc(tokens_out)

                        if self._replay_logger is not None and decision_id is not None:
                            ttft_ms = ((t_first_token - t0) * 1000.0) if t_first_token else latency_ms
                            itl_ms = ((latency_ms - ttft_ms) / max(1, tokens_out - 1)) if tokens_out > 1 else 0.0
                            self._replay_logger.log_feedback({
                                "event": "feedback",
                                "decision_id": decision_id,
                                "run_id": self._replay_logger.run_id,
                                "session_id": prefix_id,
                                "chosen_worker": chosen,
                                "ttft_ms": round(ttft_ms, 2),
                                "tokens_out": tokens_out,
                                "itl_ms": round(itl_ms, 2),
                                "duration_ms": round(latency_ms, 2),
                                "metric": round(fb["metric"], 4),
                                "baseline_ema": round(fb["baseline_ema"], 4),
                                "reward": round(fb["reward"], 4),
                                "beta_after": fb["beta_after"],
                                "lints_posterior_mean": fb["lints_posterior_mean"],
                            })

                        return

            elif self.routing_mode == "kv_native" and self.kv_router is not None:
                # ---- Native KvRouter path (Phase 1 baseline) ----
                t_routing_start = time.perf_counter()
                stop_conditions = raw.get("stop_conditions")
                sampling_options = raw.get("sampling_options")

                native_decision_id = None
                native_chosen = None

                if self._replay_logger is not None:
                    native_loads = await self.kv_router.get_potential_loads(token_ids)
                    native_chosen, _, native_overlap = await self.kv_router.best_worker(token_ids)
                    native_decision_id = str(uuid.uuid4())
                    session_depth = total_requests - reuse_budget
                    native_worker_details = [{
                        "id": li["worker_id"],
                        "kv_overlap": round(
                            1.0 - li.get("potential_prefill_tokens", 0) / max(1, tokens_in), 4,
                        ),
                        "prefill_tokens": li.get("potential_prefill_tokens", 0),
                        "decode_blocks": li.get("potential_decode_blocks", 0),
                        "beta_sample": None,
                        "lints_sample": None,
                        "final_score": None,
                    } for li in native_loads]
                    self._replay_logger.log_decision({
                        "event": "route",
                        "decision_id": native_decision_id,
                        "run_id": self._replay_logger.run_id,
                        "session_id": prefix_id,
                        "llm_call_idx": session_depth - 1,
                        "session_depth": session_depth,
                        "timestamp_ns": time.time_ns(),
                        "chosen_worker": native_chosen,
                        "native_recommendation": native_chosen,
                        "overrode_native": False,
                        "features": None,
                        "workers": native_worker_details,
                    })

                stream = await self.kv_router.generate(
                    token_ids=token_ids,
                    model=self.model_name,
                    stop_conditions=stop_conditions,
                    sampling_options=sampling_options,
                )
                routing_ms = (time.perf_counter() - t_routing_start) * 1000.0
                proc_overhead_ms = (time.perf_counter() - t_proc_start) * 1000.0

                logger.info(
                    "KvRouter routing: prefix=%s routing_ms=%.1f proc_overhead_ms=%.1f",
                    prefix_id, routing_ms, proc_overhead_ms,
                )

                t0 = time.perf_counter()
                t_first_token = None
                tokens_out = 0
                async for chunk in stream:
                    if isinstance(chunk, dict):
                        data = chunk
                    else:
                        data = chunk.data() if hasattr(chunk, "data") else chunk

                    if "token_ids" in data and isinstance(data["token_ids"], list):
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        tokens_out += len(data["token_ids"])

                    yield data

                    if "finish_reason" in data and data["finish_reason"] is not None:
                        latency_seconds = time.perf_counter() - t0
                        self._metrics.request_latency_seconds.observe(latency_seconds)
                        self._metrics.tokens_in_total.inc(tokens_in)
                        self._metrics.tokens_out_total.inc(tokens_out)

                        if self._replay_logger is not None and native_decision_id is not None:
                            latency_ms = latency_seconds * 1000.0
                            ttft_ms = ((t_first_token - t0) * 1000.0) if t_first_token else latency_ms
                            itl_ms = ((latency_ms - ttft_ms) / max(1, tokens_out - 1)) if tokens_out > 1 else 0.0
                            from learners import LatencyTracker as LT
                            fb_metric, _ = LT.latency_metric(latency_ms, tokens_out)
                            self._replay_logger.log_feedback({
                                "event": "feedback",
                                "decision_id": native_decision_id,
                                "run_id": self._replay_logger.run_id,
                                "session_id": prefix_id,
                                "chosen_worker": native_chosen,
                                "ttft_ms": round(ttft_ms, 2),
                                "tokens_out": tokens_out,
                                "itl_ms": round(itl_ms, 2),
                                "duration_ms": round(latency_ms, 2),
                                "metric": round(fb_metric, 4),
                                "baseline_ema": None,
                                "reward": None,
                                "beta_after": None,
                                "lints_posterior_mean": None,
                            })

                        return

            else:
                raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        finally:
            self._metrics.active_requests.dec()


# -------------------------- worker entry point -------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the processor."""
    parser = argparse.ArgumentParser(description="Optimized Thompson Sampling Processor")
    parser.add_argument(
        "--enable-router",
        action="store_true",
        default=True,
        help="Enable router integration",
    )
    parser.add_argument(
        "--no-router",
        action="store_false",
        dest="enable_router",
        help="Disable router (use engine load balancing only)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory (for loading tokenizer and model card)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Served model name (must match frontend's --model-name)",
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=int(os.environ.get("DYNAMO_KV_BLOCK_SIZE", "64")),
        help="KV cache block size for model card registration "
        "(default: DYNAMO_KV_BLOCK_SIZE env var or 64)",
    )
    return parser.parse_args()


@dynamo_worker()  # Dynamic mode - required to call router/workers which are also dynamic
async def worker(runtime: DistributedRuntime):
    """
    Main worker entry point for the Thompson Sampling processor.

    This processor registers as a backend that the frontend can discover via ETCD,
    then forwards requests to actual workers after applying Thompson Sampling routing.
    """
    args = parse_args()

    # Read router_type from config.yaml to determine routing mode.
    # Valid values: "kv_native" (baseline) or "kv_thompson" (Thompson + native KvRouter lifecycle).
    config_path = os.environ.get("ROUTER_CONFIG_PATH", "/workspace/custom_dynamo/config.yaml")
    routing_mode = "kv_thompson"
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        router_type = config.get("infrastructure", {}).get("router_type", "kv_thompson")
        if router_type not in ("kv_native", "kv_thompson"):
            logger.warning("Unknown router_type=%s in %s, defaulting to kv_thompson", router_type, config_path)
            router_type = "kv_thompson"
        routing_mode = router_type
        logger.info("Config %s: router_type=%s → routing_mode=%s", config_path, router_type, routing_mode)
    except Exception as e:
        logger.warning("Could not read config at %s (%s), defaulting to kv_thompson", config_path, e)

    # DYNAMIC DISCOVERY MODE:
    # Instead of using --static-endpoint on the frontend, we register a model card
    # in ETCD so the frontend can discover us via its ModelWatcher.
    #
    # This is the forward-compatible approach since --static-endpoint is deprecated.
    #
    # Flow:
    #   1. We register as dynamo.backend.generate (dynamically with instance ID)
    #   2. We call register_llm() to advertise ourselves in ETCD
    #   3. Frontend's ModelWatcher discovers us and routes requests to us
    #   4. We forward to actual workers at workers.worker.generate

    endpoint = _get_endpoint(runtime, "dynamo", "backend", "generate")

    # Register the model card with ETCD so the frontend can discover us
    # We accept preprocessed tokens (ModelInput.Tokens) and serve chat/completions
    logger.info(
        "Registering model card: model_name=%s, model_path=%s",
        args.model_name,
        args.model_path,
    )
    # IMPORTANT: kv_cache_block_size must match what workers use so checksums agree
    # and the frontend accepts this processor's model card.
    await register_llm(
        model_input=ModelInput.Tokens,  # We accept tokenized input from frontend
        model_type=ModelType.Chat | ModelType.Completions,  # Chat and completions endpoints
        endpoint=endpoint,
        model_path=args.model_path,
        model_name=args.model_name,
        kv_cache_block_size=args.kv_cache_block_size,
    )
    logger.info("Model card registered successfully - frontend can now discover us via ETCD")

    # Initialize the request handler with the endpoint for metrics
    handler = ProcessorRequestHandler(
        runtime=runtime,
        endpoint=endpoint,
        enable_router=args.enable_router,
        routing_mode=routing_mode,
        model_name=args.model_name,
        kv_block_size=args.kv_cache_block_size,
    )
    await handler.initialize()

    # Start router management HTTP server (for state persistence, config hot-reload, reset)
    if handler.thompson_router is not None:
        from router import RouterManagementServer
        mgmt_port = int(os.environ.get("LEARNER_STATE_PORT", "8084"))
        mgmt_server = RouterManagementServer(handler.thompson_router, port=mgmt_port)
        await mgmt_server.start()

    # Serve as "backend.generate" - frontend will route to us after ETCD discovery
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter
