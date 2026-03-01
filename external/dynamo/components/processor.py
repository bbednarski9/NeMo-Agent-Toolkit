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
        → queries Thompson Sampling router → worker_id
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
import json
import logging
import logging.handlers
import os
import time
import uuid
from collections.abc import AsyncIterator
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
from pydantic import BaseModel

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ----------------------- overhead file logger ----------------------- #
def _setup_overhead_logger(log_path: str = "/tmp/processor_overhead.jsonl") -> logging.Logger:
    """
    Set up a dedicated JSON Lines file logger for NATS / routing overhead.

    Each record is a single JSON object on one line.  Load for analysis with:

        import pandas as pd
        df = pd.read_json("/tmp/processor_overhead.jsonl", lines=True)

    The logger rotates at 50 MB and keeps 5 backups so it never fills disk.
    """
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    oh_logger = logging.getLogger("processor.overhead")
    oh_logger.setLevel(logging.DEBUG)
    oh_logger.propagate = False  # don't double-emit to the root Dynamo logger

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    # Formatter emits the raw message only — each call passes a pre-built JSON string.
    handler.setFormatter(logging.Formatter("%(message)s"))
    oh_logger.addHandler(handler)

    logger.info("Overhead log → %s  (max 5 × 50 MB, JSON Lines format)", log_path)
    return oh_logger


# ----------------------- request / response models ----------------------- #
class RouterRequest(BaseModel):
    """Request to the Thompson Sampling router."""

    tokens: list[int]
    prefix_id: str = "<no_reuse>"
    reuse_budget: int = 0  # remaining *after this request*
    expected_osl: int = 250
    interarrival: int = 250


class RouterFeedbackRequest(BaseModel):
    """Feedback to the router after request completion."""

    decision_id: str
    latency_ms: float
    success: bool | None = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


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
    Thompson Sampling router or native KvRouter for worker selection.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        endpoint,
        enable_router: bool = True,
        routing_mode: str = "thompson",
        model_name: str = "",
        kv_block_size: int = 16,
    ):
        """
        Initialize the processor request handler.

        Args:
            runtime: Dynamo distributed runtime for client connections.
            endpoint: Dynamo endpoint for metrics registration.
            enable_router: Whether to use any router (default: True).
            routing_mode: "thompson" (custom router via NATS RPC) or
                          "kv_native" (Dynamo's native KvRouter in-process).
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
        self.router_pick_client = None
        self.router_feedback_client = None
        self.engine_client = None
        self.kv_router = None  # Native KvRouter (Phase 1 & 2)

        # Thompson learners for kv_thompson_native mode (Phase 2)
        self.beta_learner = None
        self.lints_learner = None
        self.latency_tracker = None
        self._prefix_workers: dict[str, int] = {}  # prefix_id → last worker_id

        # Prefix-level state: {prefix_id: {"total": int, "processed": int}}
        self._prefix_state: dict[str, dict[str, int]] = {}
        self._prefix_lock = asyncio.Lock()

        # Prevent fire-and-forget tasks from being garbage-collected
        self._background_tasks: set[asyncio.Task] = set()

        # Metrics and overhead logger (initialized in initialize())
        self._metrics: ProcessorMetrics | None = None
        self._overhead_log: logging.Logger | None = None

    async def initialize(self):
        """Initialize processor by setting up metrics and connecting to services."""
        # Initialize metrics using Dynamo's metrics API
        self._metrics = ProcessorMetrics(self.endpoint)

        # Set up dedicated JSON Lines file for overhead / routing timing analysis.
        # Path can be overridden via PROCESSOR_OVERHEAD_LOG env var.
        overhead_log_path = os.environ.get("PROCESSOR_OVERHEAD_LOG", "/tmp/processor_overhead.jsonl")
        self._overhead_log = _setup_overhead_logger(overhead_log_path)

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

        if self.enable_router and self.routing_mode in ("kv_native", "kv_thompson_native"):
            # Use Dynamo's native KvRouter in-process.
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
                logger.error(
                    "KvRouter not available — requires source-built Dynamo image. "
                    "Falling back to Thompson routing."
                )
                self.routing_mode = "thompson"

        if self.enable_router and self.routing_mode == "kv_thompson_native":
            # Phase 2: Thompson learners running in-process alongside KvRouter.
            from learners import BetaLearner, LatencyTracker, LinTSLearner
            self.beta_learner = BetaLearner(decay=0.995)
            self.lints_learner = LinTSLearner(
                feature_dim=6, lambda_=1.0, v=0.25, forget_rate=0.995,
            )
            self.latency_tracker = LatencyTracker(ema_alpha=0.2)
            logger.info("Thompson learners initialized (beta_decay=0.995, lints_dim=6, mode=kv_thompson_native)")

        if self.enable_router and self.routing_mode == "thompson":
            # Thompson Sampling router via NATS RPC
            router_find_ep = _get_endpoint(self.runtime, "dynamo", "router", "find_worker")
            router_fb_ep = _get_endpoint(self.runtime, "dynamo", "router", "feedback")
            self.router_pick_client = await router_find_ep.client()
            self.router_feedback_client = await router_fb_ep.client()
            logger.info("Thompson router clients created, waiting for instances...")
            await self.router_pick_client.wait_for_instances()
            logger.info("Thompson router clients initialized successfully")

        logger.info("Processor initialized (routing_mode=%s, workers=%s/generate)",
                    self.routing_mode, worker_component_name)

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

    async def _pick_worker(
        self,
        token_ids: list[int],
        prefix_id: str,
        reuse_budget: int,
        osl: int,
        iat: int,
    ) -> tuple[int | None, str | None]:
        """
        Pick a worker via the Thompson Sampling router.

        Returns: (worker_id, decision_id) or (None, None) if routing fails.
        """
        if not self.router_pick_client:
            return None, None

        req = RouterRequest(
            tokens=token_ids,
            prefix_id=prefix_id,
            reuse_budget=max(int(reuse_budget), 0),
            expected_osl=osl,
            interarrival=iat,
        )

        try:
            t_nats_start = time.perf_counter()
            stream = await self.router_pick_client.generate(req.model_dump())

            worker_id: int | None = None
            decision_id: str | None = None

            async for chunk in stream:
                data = chunk.data()
                if "error" in data:
                    logger.error("Router error: %s", data["error"])
                    self._metrics.router_errors_total.inc()
                    break

                wid = data.get("worker_id", -1)
                if wid == -1:
                    break

                worker_id = int(wid)
                decision_id = data.get("decision_id")
                break

            nats_rtt_ms = (time.perf_counter() - t_nats_start) * 1000.0

            # Record routing decision
            if worker_id is not None:
                self._metrics.routing_decisions_total.labels(worker_id=str(worker_id)).inc()
                logger.info(
                    "nats_overhead: prefix=%s worker=%s nats_rtt_ms=%.1f",
                    prefix_id, worker_id, nats_rtt_ms,
                )
            else:
                logger.warning(
                    "nats_overhead: prefix=%s nats_rtt_ms=%.1f (no worker returned, fallback)",
                    prefix_id, nats_rtt_ms,
                )

            return worker_id, decision_id

        except Exception:
            logger.exception("Failed to pick worker")
            self._metrics.router_errors_total.inc()
            return None, None

    async def _send_feedback_safely(
        self,
        decision_id: str | None,
        latency_ms: float,
        success: bool,
        tokens_in: int,
        tokens_out: int,
        finish_reason: str | None,
    ):
        """
        Send feedback to router (fire-and-forget style).

        This feedback is used by the Thompson Sampling algorithm to update
        its model of worker performance.
        """
        if not decision_id or not self.router_feedback_client:
            return

        try:
            feedback = RouterFeedbackRequest(
                decision_id=decision_id,
                latency_ms=float(latency_ms),
                success=bool(success),
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                finish_reason=finish_reason or "",
            )
            stream = await self.router_feedback_client.generate(feedback.model_dump())
            async for _ in stream:
                pass
        except Exception:
            logger.exception("Failed to send router feedback")

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

    async def _stream_from_engine(
        self,
        request: dict[str, Any],
        worker_id: int | None,
        decision_id: str | None,
        tokens_in: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream response from the backend engine.

        Yields response chunks and sends feedback to the router on completion.
        Also updates Prometheus metrics for latency and token throughput.

        KV cache efficiency (KVE) metrics are updated asynchronously via
        create_task() to ensure zero impact on routing throughput.
        """
        t0 = time.perf_counter()
        tokens_out = 0
        finish_reason: str | None = None
        kve_data: KVEfficiencyData | None = None  # Collected from response

        try:
            # Route to specific worker or use engine's load balancing
            if worker_id is not None:
                stream = await self.engine_client.direct(request, worker_id)
            else:
                stream = await self.engine_client.generate(request)

            async for chunk in stream:
                data = chunk.data()

                # Handle engine errors
                if "error" in data:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    await self._send_feedback_safely(decision_id, latency_ms, False, tokens_in, tokens_out, "error")
                    self._metrics.engine_errors_total.inc()
                    yield {"error": data["error"]}
                    return

                # Count output tokens
                if "token_ids" in data and isinstance(data["token_ids"], list):
                    tokens_out += len(data["token_ids"])

                # Extract KVE data if present (typically in final chunk or usage chunk)
                # We check for 'usage' field which contains cache efficiency info
                if "usage" in data or "nvext" in data:
                    extracted = KVEfficiencyData.from_response(data)
                    if extracted.has_data():
                        kve_data = extracted

                # Pass through the chunk
                yield data

                # Handle completion
                if "finish_reason" in data and data["finish_reason"] is not None:
                    finish_reason = data["finish_reason"]
                    latency_seconds = time.perf_counter() - t0
                    latency_ms = latency_seconds * 1000.0

                    # Send feedback to router (fire-and-forget — don't block generator return)
                    feedback_task = asyncio.create_task(
                        self._send_feedback_safely(decision_id, latency_ms, True, tokens_in, tokens_out, finish_reason))
                    self._background_tasks.add(feedback_task)
                    feedback_task.add_done_callback(self._background_tasks.discard)

                    # Update core Prometheus metrics (fast atomic operations)
                    self._metrics.request_latency_seconds.observe(latency_seconds)
                    self._metrics.tokens_in_total.inc(tokens_in)
                    self._metrics.tokens_out_total.inc(tokens_out)

                    # Fire-and-forget KVE metric update (async, non-blocking)
                    # This ensures KVE computation has ZERO impact on routing throughput.
                    # Tasks are stored in _background_tasks to prevent garbage collection.
                    if kve_data is not None:
                        task = asyncio.create_task(self._update_kve_metrics_async(kve_data))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

                    return

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            await self._send_feedback_safely(decision_id, latency_ms, False, tokens_in, tokens_out, "exception")
            self._metrics.engine_errors_total.inc()
            logger.exception("Engine stream exception")
            yield {"error": str(e)}
            return

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

            if self.routing_mode == "kv_thompson_native" and self.kv_router is not None:
                # ---- Thompson + Native KvRouter lifecycle (Phase 2) ----
                import math
                import numpy as np

                t_routing_start = time.perf_counter()

                # 1. Get native load signals per worker
                loads = await self.kv_router.get_potential_loads(token_ids)

                # 2. Query native router for its pick + overlap (no state update)
                native_pick, _, overlap_blocks = await self.kv_router.best_worker(token_ids)

                # 3. Build per-worker scores using Thompson learners
                worker_scores: dict[int, float] = {}
                last_worker = self._prefix_workers.get(prefix_id)

                for load_info in loads:
                    wid = load_info["worker_id"]
                    prefill_tokens = load_info.get("potential_prefill_tokens", 0)
                    decode_blocks = load_info.get("potential_decode_blocks", 0)

                    # Ensure learners know this worker
                    self.beta_learner.add_worker(wid)
                    self.lints_learner.add_worker(wid)

                    # Feature vector: [bias, inv_prefill, inv_decode, affinity, osl_norm, reuse_norm]
                    inv_prefill = 1.0 / (1.0 + prefill_tokens / 1000.0)
                    inv_decode = 1.0 / (1.0 + decode_blocks / 50.0)
                    affinity = 1.0 if (last_worker is not None and wid == last_worker) else 0.0
                    osl_norm = min(osl, 1024) / 1024.0
                    reuse_norm = math.tanh(0.25 * max(reuse_budget, 0))

                    x = np.array([1.0, inv_prefill, inv_decode, affinity, osl_norm, reuse_norm],
                                 dtype=np.float64)

                    # Combined score: base (low prefill = good) + beta exploration + lints contextual
                    base_score = inv_prefill * 0.5 + inv_decode * 0.3
                    beta_s = 0.05 * self.beta_learner.sample(wid)
                    lints_s = math.tanh(self.lints_learner.sample(wid, x))

                    worker_scores[wid] = base_score + beta_s + lints_s

                # 4. Pick best-scoring worker
                chosen = max(worker_scores, key=worker_scores.get) if worker_scores else native_pick
                self._prefix_workers[prefix_id] = chosen

                routing_ms = (time.perf_counter() - t_routing_start) * 1000.0
                proc_overhead_ms = (time.perf_counter() - t_proc_start) * 1000.0

                logger.info(
                    "ThompsonNative routing: prefix=%s chosen=%s native_pick=%s "
                    "workers=%d routing_ms=%.1f",
                    prefix_id, chosen, native_pick, len(worker_scores), routing_ms,
                )

                # 5. Route via KvRouter with forced worker_id (handles lifecycle)
                stop_conditions = raw.get("stop_conditions")
                sampling_options = raw.get("sampling_options")

                stream = await self.kv_router.generate(
                    token_ids=token_ids,
                    model=self.model_name,
                    stop_conditions=stop_conditions,
                    sampling_options=sampling_options,
                    worker_id=chosen,
                )

                # 6. Stream response and collect feedback
                t0 = time.perf_counter()
                tokens_out = 0
                async for chunk in stream:
                    if isinstance(chunk, dict):
                        data = chunk
                    else:
                        data = chunk.data() if hasattr(chunk, "data") else chunk

                    if "token_ids" in data and isinstance(data["token_ids"], list):
                        tokens_out += len(data["token_ids"])

                    yield data

                    if "finish_reason" in data and data["finish_reason"] is not None:
                        latency_seconds = time.perf_counter() - t0
                        latency_ms = latency_seconds * 1000.0

                        # 7. Update Thompson learners with observed reward
                        from learners import LatencyTracker as LT
                        metric, per_tok = LT.latency_metric(latency_ms, tokens_out)
                        baseline = self.latency_tracker.get_global_baseline(per_tok, fallback=metric)
                        reward = LT.compute_reward(metric, baseline, True)
                        self.beta_learner.update(chosen, reward)

                        x_chosen = np.array([1.0,
                                             1.0 / (1.0 + prefill_tokens / 1000.0),
                                             1.0 / (1.0 + decode_blocks / 50.0),
                                             1.0 if last_worker == chosen else 0.0,
                                             min(osl, 1024) / 1024.0,
                                             math.tanh(0.25 * max(reuse_budget, 0))],
                                            dtype=np.float64)
                        self.lints_learner.update(chosen, x_chosen, reward)
                        self.latency_tracker.update_baselines(chosen, "M", "L", metric, per_tok)

                        self._metrics.request_latency_seconds.observe(latency_seconds)
                        self._metrics.tokens_in_total.inc(tokens_in)
                        self._metrics.tokens_out_total.inc(tokens_out)

                        logger.debug(
                            "ThompsonNative feedback: wid=%s metric=%.2f baseline=%.2f "
                            "reward=%.3f tokens_out=%d",
                            chosen, metric, baseline, reward, tokens_out,
                        )
                        return

            elif self.routing_mode == "kv_native" and self.kv_router is not None:
                # ---- Native KvRouter path (Phase 1 baseline) ----
                t_routing_start = time.perf_counter()
                stop_conditions = raw.get("stop_conditions")
                sampling_options = raw.get("sampling_options")

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
                tokens_out = 0
                async for chunk in stream:
                    if isinstance(chunk, dict):
                        data = chunk
                    else:
                        data = chunk.data() if hasattr(chunk, "data") else chunk

                    if "token_ids" in data and isinstance(data["token_ids"], list):
                        tokens_out += len(data["token_ids"])

                    yield data

                    if "finish_reason" in data and data["finish_reason"] is not None:
                        latency_seconds = time.perf_counter() - t0
                        self._metrics.request_latency_seconds.observe(latency_seconds)
                        self._metrics.tokens_in_total.inc(tokens_in)
                        self._metrics.tokens_out_total.inc(tokens_out)
                        return

            else:
                # ---- Thompson Sampling path (NATS RPC to custom router) ----
                t_routing_start = time.perf_counter()
                worker_id, decision_id = await self._pick_worker(token_ids, prefix_id, reuse_budget, osl, iat)
                routing_ms = (time.perf_counter() - t_routing_start) * 1000.0

                proc_overhead_ms = (time.perf_counter() - t_proc_start) * 1000.0

                logger.info(
                    "Routing decision: prefix=%s worker=%s decision=%s reuse_budget=%d "
                    "routing_ms=%.1f proc_overhead_ms=%.1f",
                    prefix_id,
                    worker_id,
                    decision_id,
                    reuse_budget,
                    routing_ms,
                    proc_overhead_ms,
                )

                if self._overhead_log is not None:
                    record = {
                        "ts": time.time(),
                        "prefix_id": prefix_id,
                        "worker_id": worker_id,
                        "decision_id": decision_id,
                        "tokens_in": tokens_in,
                        "osl": osl,
                        "iat": iat,
                        "reuse_budget": reuse_budget,
                        "is_auto": is_auto,
                        "nats_rtt_ms": round(routing_ms, 2),
                        "proc_overhead_ms": round(proc_overhead_ms, 2),
                        "routing_routed": worker_id is not None,
                    }
                    self._overhead_log.info(json.dumps(record))

                async for resp in self._stream_from_engine(raw, worker_id, decision_id, tokens_in):
                    yield resp

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
    # "kv_native" → in-process native KvRouter; anything else → Thompson via NATS RPC.
    config_path = os.environ.get("ROUTER_CONFIG_PATH", "/workspace/custom_dynamo/config.yaml")
    routing_mode = "thompson"
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        router_type = config.get("infrastructure", {}).get("router_type", "kv_thompson")
        if router_type in ("kv_native", "kv_thompson_native"):
            routing_mode = router_type
        logger.info("Config %s: router_type=%s → routing_mode=%s", config_path, router_type, routing_mode)
    except Exception as e:
        logger.warning("Could not read config at %s (%s), defaulting to thompson", config_path, e)

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

    # Serve as "backend.generate" - frontend will route to us after ETCD discovery
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter
