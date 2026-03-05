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
Optimized Thompson Sampling Router with Prometheus Metrics.

This router implements Contextual Thompson Sampling with:
  - KV overlap locality
  - Remaining per-prefix requests (reuse_budget)
  - OSL-based decode cost, ISL/prefill cost per worker
  - IAT-based stickiness/opportunity weighting
  - Instant & outstanding load (no TTL decay)
  - Delayed bandit update using observed latency via `feedback` endpoint
  - Timeout penalty for missing feedback
  - Prometheus metrics (instead of CSV)
  - Debug traces for offline analysis

Key differences from generalized/router.py:
  - Uses Prometheus metrics instead of CSV logging
  - Removed CSV file I/O
  - Added comprehensive Prometheus gauges, counters, and histograms
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import uvloop
import yaml
from aiohttp import web
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging


def _get_endpoint(runtime: DistributedRuntime, namespace: str, component: str, endpoint: str):
    """Compat shim: old API uses namespace().component().endpoint(), new uses endpoint("ns.comp.ep")."""
    if hasattr(runtime, "namespace"):
        return runtime.namespace(namespace).component(component).endpoint(endpoint)
    return runtime.endpoint(f"{namespace}.{component}.{endpoint}")


# KV cache overlap scoring — uses RadixTree + ZmqKvEventListener from dynamo.llm.
# Backend-agnostic: works identically with SGLang and vLLM workers.
# Falls back gracefully to empty scores if dynamo.llm primitives are unavailable.
from kv_indexer import KvIndexer
from kv_indexer import OverlapScores
from learners import BetaLearner, LatencyTracker, LinTSLearner, PendingDecisions
from pydantic import BaseModel, field_validator

configure_dynamo_logging()
logger = logging.getLogger(__name__)

WorkerId = int


# ---------------------- config loading ---------------------- #
def get_default_config_path() -> Path:
    """Get path to default config.yaml in the same directory as this script."""
    return Path(__file__).parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml.

    Returns:
        Configuration dictionary with nested structure.
    """
    if config_path is None:
        config_path = get_default_config_path()

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s, using built-in defaults", config_path)
        return get_builtin_defaults()

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from: %s", config_path)
    return config


def get_builtin_defaults() -> dict[str, Any]:
    """Return built-in default configuration (matches config.yaml)."""
    return {
        "infrastructure": {
            "block_size": 64,
            "router_type": "kv",
            "min_workers": 1,
        },
        "affinity": {
            "base": 0.30,
            "reuse_weight": 0.15,
            "iat_weight": 0.20,
            "sticky_load_floor": 0.01,
        },
        "exploration": {
            "base_ts_weight": 0.10,
            "beta_decay": 1.0,
            "temperature": {
                "base": 1.0,
                "min": 0.15,
                "max": 2.0,
            },
        },
        "switching_cost": {
            "base": 0.20,
            "reuse_penalty": 0.08,
            "iat_penalty": 0.05,
        },
        "load_balancing": {
            "queue_penalty_weight": 0.50,
            "gpu_penalty_weight": 1.00,
            "outstanding_work_weight": 0.45,
            "job_gpu_coupling_weight": 0.40,
            "job_queue_coupling_weight": 0.20,
        },
        "prefill": {
            "token_scale": 1024.0,
            "weight": 1.0,
        },
        "lints": {
            "lambda": 1.0,
            "v": 0.25,
            "forget_rate": 0.995,
        },
        "feedback": {
            "timeout_seconds": 120.0,
            "sweep_interval_seconds": 5.0,
            "timeout_reward": 0.0,
            "latency_ema_alpha": 0.2,
            "reward_baseline_mode": "global",
        },
        "debug": {
            "traces_enabled": False,
            "trace_dir": "/tmp/dynamo_router_traces",
            "buffer_size": 2000,
        },
    }


def get_nested(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a nested value from config using dot notation.

    Args:
        config: Configuration dictionary
        dotted_key: Key in dot notation, e.g., "affinity.base"
        default: Default value if key not found

    Returns:
        Value at the nested key, or default if not found.
    """
    keys = dotted_key.split(".")
    obj = config
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set a nested value in config using dot notation.

    Args:
        config: Configuration dictionary (modified in place)
        dotted_key: Key in dot notation, e.g., "affinity.base"
        value: Value to set
    """
    keys = dotted_key.split(".")
    obj = config
    for k in keys[:-1]:
        if k not in obj:
            obj[k] = {}
        obj = obj[k]
    obj[keys[-1]] = value


def auto_cast(value_str: str) -> Any:
    """Auto-cast a string value to appropriate type.

    Args:
        value_str: String value from CLI

    Returns:
        Value cast to int, float, bool, or str as appropriate.
    """
    # Boolean
    if value_str.lower() in ("true", "yes", "1"):
        return True
    if value_str.lower() in ("false", "no", "0"):
        return False

    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String
    return value_str


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to configuration.

    Args:
        config: Base configuration dictionary
        args: Parsed CLI arguments

    Returns:
        Configuration with CLI overrides applied.
    """
    # Apply explicit CLI flags
    if args.affinity_base is not None:
        set_nested(config, "affinity.base", args.affinity_base)
        logger.info("CLI override: affinity.base = %s", args.affinity_base)

    if args.temp_base is not None:
        set_nested(config, "exploration.temperature.base", args.temp_base)
        logger.info("CLI override: exploration.temperature.base = %s", args.temp_base)

    if args.lints_v is not None:
        set_nested(config, "lints.v", args.lints_v)
        logger.info("CLI override: lints.v = %s", args.lints_v)

    # Apply generic --override flags
    if args.override:
        for override in args.override:
            if "=" not in override:
                logger.warning("Invalid override format (expected key=value): %s", override)
                continue
            key, value_str = override.split("=", 1)
            value = auto_cast(value_str)
            set_nested(config, key, value)
            logger.info("CLI override: %s = %s", key, value)

    return config


def _init_prometheus_metrics():
    """Initialize Prometheus metrics lazily."""
    import functools

    @functools.lru_cache(maxsize=1)
    def _init() -> dict:
        metrics: dict = {}
        try:
            from prometheus_client import REGISTRY
            from prometheus_client import Counter
            from prometheus_client import Gauge
            from prometheus_client import Histogram

            metrics["decisions_total"] = Counter(
                "thompson_router_decisions_total",
                "Total routing decisions by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["kv_overlap"] = Gauge(
                "thompson_router_kv_overlap",
                "KV cache overlap score for last decision by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["feedback_latency"] = Histogram(
                "thompson_router_feedback_latency_seconds",
                "Latency from feedback by worker",
                ["worker_id"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
                registry=REGISTRY,
            )
            metrics["reward"] = Gauge(
                "thompson_router_reward",
                "Last computed reward by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["pending_decisions"] = Gauge(
                "thompson_router_pending_decisions",
                "Number of pending decisions awaiting feedback",
                registry=REGISTRY,
            )
            metrics["timeout_penalties"] = Counter(
                "thompson_router_timeout_penalties_total",
                "Total timeout penalties applied",
                registry=REGISTRY,
            )
            metrics["sticky_decisions"] = Counter(
                "thompson_router_sticky_decisions_total",
                "Decisions that stayed on the same worker (sticky)",
                registry=REGISTRY,
            )
            metrics["switch_decisions"] = Counter(
                "thompson_router_switch_decisions_total",
                "Decisions that switched to a different worker",
                registry=REGISTRY,
            )
            metrics["beta_alpha"] = Gauge(
                "thompson_router_beta_alpha",
                "Beta distribution alpha parameter by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["beta_beta"] = Gauge(
                "thompson_router_beta_beta",
                "Beta distribution beta parameter by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["prefix_state_size"] = Gauge(
                "thompson_router_prefix_state_size",
                "Number of active prefix states",
                registry=REGISTRY,
            )
            metrics["reuse_budget"] = Histogram(
                "thompson_router_reuse_budget",
                "Distribution of reuse_budget values",
                buckets=[0, 1, 2, 5, 10, 20, 50, 100],
                registry=REGISTRY,
            )
            metrics["tokens_per_request"] = Histogram(
                "thompson_router_tokens_per_request",
                "Distribution of input token counts",
                buckets=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                registry=REGISTRY,
            )
            metrics["decisions_by_domain"] = Counter(
                "thompson_router_decisions_by_domain_total",
                "Routing decisions by worker and domain",
                ["worker_id", "domain"],
                registry=REGISTRY,
            )
            # KV opportunity-cost metrics: measure the trade-off between KV
            # cache hit rate and worker load at each routing decision.
            metrics["kv_vs_idle_delta"] = Histogram(
                "thompson_router_kv_vs_idle_delta",
                "KV overlap delta: chosen_overlap - best_idle_overlap.  "
                "Positive = routed to a better KV match but busier worker.  "
                "Negative = routed to idle worker with worse KV coverage.",
                buckets=[-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
                registry=REGISTRY,
            )
            metrics["routed_to_queued"] = Counter(
                "thompson_router_routed_to_queued_total",
                "Decisions where chosen worker had queue depth > 0 and an idle alternative existed",
                registry=REGISTRY,
            )
            metrics["idle_kv_miss"] = Histogram(
                "thompson_router_idle_kv_miss",
                "KV overlap of the BEST IDLE worker at routing time "
                "(shows what KV coverage would be if we always routed to the lowest-queue worker).",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
                registry=REGISTRY,
            )
            logger.info("Prometheus metrics initialized for router")
        except ImportError:
            logger.warning("prometheus_client not available, metrics disabled")

        return metrics

    return _init()


# ---------------------- request / response models ---------------------- #
_OSL_CAT_TO_INT: dict[str, int] = {"LOW": 128, "MEDIUM": 250, "HIGH": 1024}
_IAT_CAT_TO_INT: dict[str, int] = {"LOW": 50, "MEDIUM": 250, "HIGH": 1000}


def _cat_or_int(value: str | int | None, cat_lut: dict[str, int], default: int) -> int:
    """Accept int pass-through or convert a category string to its int value."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().upper()
    return cat_lut.get(s, default)


class RouterRequest(BaseModel):
    tokens: list[int]
    prefix_id: str = "<no_reuse>"
    reuse_budget: int = 0  # remaining *after this request*
    expected_osl: int = 250
    interarrival: int = 250

    @field_validator("expected_osl", mode="before")
    @classmethod
    def _normalise_osl(cls, v: str | int | None) -> int:
        return _cat_or_int(v, _OSL_CAT_TO_INT, default=250)

    @field_validator("interarrival", mode="before")
    @classmethod
    def _normalise_iat(cls, v: str | int | None) -> int:
        return _cat_or_int(v, _IAT_CAT_TO_INT, default=250)


class RouterResponse(BaseModel):
    worker_id: int
    prefix_hit_rate: float
    decision_id: str | None = None


class FeedbackRequest(BaseModel):
    decision_id: str
    latency_ms: float
    success: bool | None = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


class FeedbackAck(BaseModel):
    ok: bool
    used_baseline: float
    reward: float
    worker_id: int | None = None
    error: str | None = None


# ---------------------- helper decorator ---------------------- #
def safe_update(lock_name: str):

    def decorator(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_name)
            with lock:
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


# ---------------------- router implementation ---------------------- #
class WorkloadAwareRouter:
    """
    Contextual Thompson Sampling router with Prometheus metrics.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        block_size: int = 64,
        router_type: str = "kv",
        min_workers: int = 1,
        # Affinity / exploration
        affinity_base: float = 0.30,
        affinity_reuse_weight: float = 0.15,
        affinity_iat_weight: float = 0.20,
        base_ts_weight: float = 0.10,
        sticky_load_floor: float = 0.70,
        # Softmax temperature
        temp_base: float = 1.0,
        temp_min: float = 0.15,
        temp_max: float = 2.0,
        # Switching cost
        switch_cost_base: float = 0.20,
        switch_cost_reuse: float = 0.08,
        switch_cost_iat: float = 0.05,
        # Load / opportunity cost
        queue_penalty_weight: float = 0.50,
        gpu_penalty_weight: float = 1.00,
        outstanding_work_weight: float = 0.45,
        job_gpu_coupling_weight: float = 0.40,
        job_queue_coupling_weight: float = 0.20,
        # Prefill / ISL
        prefill_token_scale: float = 1024.0,
        prefill_weight: float = 1.0,
        # LinTS
        lints_lambda: float = 1.0,
        lints_v: float = 0.25,
        lints_forget: float = 0.995,
        # Beta bandit
        beta_decay: float = 1.0,
        # ---------- Feedback timeout / sweep ----------
        feedback_timeout_seconds: float = 120.0,
        pending_sweep_interval_seconds: float = 5.0,
        timeout_reward: float = 0.0,
        # ---------- Latency EMA (reward normalization) ----------
        latency_ema_alpha: float = 0.2,
        reward_baseline_mode: str = "global",
        # ---------- kv_load (Dynamo-native formula) ----------
        kv_load_overlap_weight: float = 1.0,
        kv_load_temperature: float = 0.0,
        metrics_scrape_interval: float = 0.1,
        # ---------- kv_load_balanced / kv_thompson ----------
        idle_boost: float = 0.1,
        kv_load_temp: float = 0.5,
        kv_ts_weight: float = 0.05,
        kv_thompson_queue_penalty_weight: float | None = None,
        kv_thompson_cold_start: float = 0.05,
        # ---------- kv_thompson feature toggles ----------
        kt_enable_lints: bool = False,
        kt_enable_affinity: bool = False,
        kt_enable_switching_cost: bool = False,
        kt_enable_full_load: bool = False,
        kt_enable_adaptive_temp: bool = False,
        kt_enable_adaptive_explore: bool = False,
        kt_enable_sticky_floor: bool = False,
        kt_lints_weight: float = 1.0,
        kt_affinity_base: float = 0.30,
        kt_affinity_reuse_weight: float = 0.08,
        kt_affinity_iat_weight: float = 0.20,
        kt_switch_base: float = 0.10,
        kt_switch_reuse: float = 0.04,
        kt_switch_iat: float = 0.03,
        kt_sticky_load_floor: float = 0.01,
        kt_load_mod_floor: float = 0.0,
        kt_adaptive_temp_base: float = 1.0,
        # ---------- Debug traces ----------
        debug_traces: bool = False,
        debug_trace_dir: str = "/tmp/dynamo_router_traces",
        debug_buffer_size: int = 2000,
    ):
        self.runtime = runtime
        self.block_size = block_size
        self.router_type = router_type
        self.min_workers = min_workers

        # clients / helpers (initialized later)
        self.engine_client = None
        self.indexer: KvIndexer | None = None

        # concurrency primitives (learner-specific locks are inside learner objects)
        self._init_lock = threading.Lock()
        self._prefix_lock = threading.Lock()

        # prefix state: pid -> {"worker": int|None, "reuse_remaining": int}
        self.prefix_cache_state: dict[str, dict[str, int | None]] = {}
        # pid -> {"decode_cost","prefill_cost","iat_factor"}
        self.prefix_meta: dict[str, dict[str, float]] = {}

        # Modular learner components
        self.feature_dim = 9
        self.beta_learner = BetaLearner(decay=float(beta_decay))
        self.lints_learner = LinTSLearner(
            feature_dim=self.feature_dim,
            lambda_=float(lints_lambda),
            v=float(lints_v),
            forget_rate=float(lints_forget),
        )
        self.latency_tracker = LatencyTracker(ema_alpha=float(latency_ema_alpha))
        self.reward_baseline_mode = str(reward_baseline_mode)
        # Kept for backward-compat references inside router
        self.lin_lambda = self.lints_learner.lambda_
        self.lin_v = self.lints_learner.v
        self.lin_forget = self.lints_learner.forget_rate

        # knobs
        self.affinity_base = float(affinity_base)
        self.affinity_reuse_weight = float(affinity_reuse_weight)
        self.affinity_iat_weight = float(affinity_iat_weight)
        self.base_ts_weight = float(base_ts_weight)
        self.sticky_load_floor = float(sticky_load_floor)
        self.temp_base = float(temp_base)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.switch_cost_base = float(switch_cost_base)
        self.switch_cost_reuse = float(switch_cost_reuse)
        self.switch_cost_iat = float(switch_cost_iat)
        self.queue_penalty_weight = float(queue_penalty_weight)
        self.gpu_penalty_weight = float(gpu_penalty_weight)
        self.outstanding_work_weight = float(outstanding_work_weight)
        self.job_gpu_coupling_weight = float(job_gpu_coupling_weight)
        self.job_queue_coupling_weight = float(job_queue_coupling_weight)
        self.prefill_token_scale = float(prefill_token_scale)
        self.prefill_weight = float(prefill_weight)

        # Feedback timeout / sweep
        self.feedback_timeout_seconds = float(feedback_timeout_seconds)
        self.pending_sweep_interval_seconds = float(pending_sweep_interval_seconds)
        self.timeout_reward = float(max(0.0, min(1.0, timeout_reward)))
        self.pending_decisions = PendingDecisions(
            timeout_seconds=self.feedback_timeout_seconds,
            sweep_interval_seconds=self.pending_sweep_interval_seconds,
        )

        # Backward-compat alias — existing code referencing self.pending
        # now goes through the PendingDecisions wrapper.
        self.pending = self.pending_decisions._pending

        # Cold-start round-robin counter for kv_only / kv_load_balanced modes
        self._cold_start_rr: int = 0

        # kv_load (Dynamo-native) parameters
        self.kv_load_overlap_weight = float(kv_load_overlap_weight)
        self.kv_load_temperature = float(kv_load_temperature)
        self.metrics_scrape_interval = float(metrics_scrape_interval)

        # kv_load: per-worker routing counter — mirrors Dynamo's
        # ActiveSequences.  Incremented on each routing decision so the
        # next decision sees the load we just created.
        self._kv_load_routed: dict[int, int] = {}  # worker_id -> cumulative routes
        self._kv_load_routed_lock = threading.Lock()

        # kv_load_balanced / kv_thompson mode parameters
        self.idle_boost = float(idle_boost)
        self.kv_load_temp = float(kv_load_temp)
        self.kv_ts_weight = float(kv_ts_weight)
        self.kv_thompson_queue_pw = kv_thompson_queue_penalty_weight
        self.kv_thompson_cold_start = float(kv_thompson_cold_start)

        # kv_thompson feature toggles
        self.kt_enable_lints = bool(kt_enable_lints)
        self.kt_enable_affinity = bool(kt_enable_affinity)
        self.kt_enable_switching_cost = bool(kt_enable_switching_cost)
        self.kt_enable_full_load = bool(kt_enable_full_load)
        self.kt_enable_adaptive_temp = bool(kt_enable_adaptive_temp)
        self.kt_enable_adaptive_explore = bool(kt_enable_adaptive_explore)
        self.kt_enable_sticky_floor = bool(kt_enable_sticky_floor)
        self.kt_lints_weight = float(kt_lints_weight)
        self.kt_affinity_base = float(kt_affinity_base)
        self.kt_affinity_reuse_weight = float(kt_affinity_reuse_weight)
        self.kt_affinity_iat_weight = float(kt_affinity_iat_weight)
        self.kt_switch_base = float(kt_switch_base)
        self.kt_switch_reuse = float(kt_switch_reuse)
        self.kt_switch_iat = float(kt_switch_iat)
        self.kt_sticky_load_floor = float(kt_sticky_load_floor)
        self.kt_load_mod_floor = float(kt_load_mod_floor)
        self.kt_adaptive_temp_base = float(kt_adaptive_temp_base)

        # Debug traces
        self.debug_traces = bool(debug_traces)
        self.debug_trace_dir = str(debug_trace_dir)
        self.recent_traces: deque = deque(maxlen=int(debug_buffer_size))
        if self.debug_traces:
            os.makedirs(self.debug_trace_dir, exist_ok=True)
            logger.info("Router debug traces enabled -> %s", self.debug_trace_dir)

        # Prometheus metrics
        self._metrics = {}

    # --------------------- tracing --------------------- #
    def _emit_trace(self, kind: str, payload: dict[str, Any]):
        if not self.debug_traces:
            return
        item = {"ts": time.time(), "kind": kind, **payload}
        self.recent_traces.append(item)
        try:
            path = os.path.join(self.debug_trace_dir, "router_traces.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.debug("Trace write failed: %s", e)

    # --------------------- level mappings (continuous) --------------------- #
    @staticmethod
    def _decode_cost(osl: int) -> float:
        """Linearly interpolate decode cost from continuous OSL (tokens).

        Anchor points: 128 -> 1.0, 250 -> 2.0, 1024 -> 3.0
        Values outside the range are clamped.
        """
        if osl <= 128:
            return 1.0
        if osl <= 250:
            return 1.0 + (osl - 128) / (250 - 128)
        if osl >= 1024:
            return 3.0
        return 2.0 + (osl - 250) / (1024 - 250)

    @staticmethod
    def _iat_factor(iat: int) -> float:
        """Linearly interpolate IAT factor from continuous IAT (ms).

        Anchor points: 50 -> 1.5, 250 -> 1.0, 1000 -> 0.6
        Values outside the range are clamped.
        """
        if iat <= 50:
            return 1.5
        if iat <= 250:
            return 1.5 - 0.5 * (iat - 50) / (250 - 50)
        if iat >= 1000:
            return 0.6
        return 1.0 - 0.4 * (iat - 250) / (1000 - 250)

    @staticmethod
    def _osl_bin(osl: int) -> str:
        """Bucket continuous OSL into LOW/MEDIUM/HIGH for latency baseline keys."""
        if osl <= 189:
            return "LOW"
        if osl <= 637:
            return "MEDIUM"
        return "HIGH"

    # --------------------- init --------------------- #
    async def initialize(self):
        """Initialize router by polling for backend workers."""
        # Initialize Prometheus metrics
        self._metrics = _init_prometheus_metrics()

        # Connect to actual workers at workers.{component}.generate
        # Workers are in the "workers" namespace (hidden from frontend discovery)
        # Component name varies by backend (REQUIRED - no default):
        #   - SGLang: uses "worker" (set via --endpoint workers.worker.generate)
        #   - vLLM: uses "backend" (hardcoded in dynamo.vllm)
        worker_component = os.environ.get("DYNAMO_WORKER_COMPONENT")
        if not worker_component:
            raise ValueError("DYNAMO_WORKER_COMPONENT environment variable is required. "
                             "Set to 'worker' for SGLang or 'backend' for vLLM.")
        engine_ep = _get_endpoint(self.runtime, "workers", worker_component, "generate")
        logger.info("Getting engine client for workers/%s/generate", worker_component)
        self.engine_client = await engine_ep.client()

        min_workers = int(self.min_workers)
        if min_workers < 0:
            raise ValueError(f"min_workers must be >= 0, got {min_workers}")

        timeout_s = float(os.environ.get("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S", "600"))
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S must be a finite number > 0")

        deadline = time.monotonic() + timeout_s
        backoff_s = 0.5

        logger.info("Waiting for backend workers (min_workers=%d, timeout_s=%.1f)...", min_workers, timeout_s)

        if min_workers == 0:
            instance_ids_raw = list(self.engine_client.instance_ids())
            logger.info("Backend workers discovered (min_workers=0): %s", instance_ids_raw)
        else:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out after {timeout_s}s waiting for >= {min_workers} backend worker(s)")

                try:
                    await asyncio.wait_for(
                        self.engine_client.wait_for_instances(),
                        timeout=min(remaining, 10.0),
                    )
                except TimeoutError:
                    pass

                instance_ids_raw = list(self.engine_client.instance_ids())
                if len(instance_ids_raw) >= min_workers:
                    try:
                        instance_ids = [int(w) for w in instance_ids_raw]
                    except Exception:
                        instance_ids = instance_ids_raw
                    logger.info("Backend workers discovered: %s", instance_ids)
                    break

                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.5, 5.0)

        self.indexer = KvIndexer(engine_ep, self.block_size)

        # Start background metrics scraper (non-blocking HTTP scrapes in a daemon thread).
        discovered_worker_ids = sorted(int(w) for w in self.engine_client.instance_ids())
        self._start_metrics_scraper(discovered_worker_ids, interval=self.metrics_scrape_interval)

        # Register workers' ZMQ KV event streams for overlap scoring.
        # Port allocation: KV_EVENT_BASE_PORT + worker_index (sorted by instance_id).
        kv_event_base_port = int(os.environ.get("KV_EVENT_BASE_PORT", "0"))
        enable_kv_events = os.environ.get("ENABLE_KV_EVENTS", "false").lower() == "true"
        is_vllm = worker_component == "backend"
        if enable_kv_events and kv_event_base_port > 0:
            discovered_ids = sorted(int(w) for w in self.engine_client.instance_ids())
            if is_vllm:
                # vLLM publishes msgpack multipart events on raw ZMQ; use the
                # native vLLM drain which handles stored + removed + cleared.
                for idx, wid in enumerate(discovered_ids):
                    endpoint = f"tcp://127.0.0.1:{kv_event_base_port + idx}"
                    self.indexer.add_vllm_worker(wid, endpoint)
                self.indexer.start_vllm_drain(interval=0.1)
                logger.info(
                    "KvIndexer: %d vLLM workers registered, event drain started (base_port=%d)",
                    len(discovered_ids),
                    kv_event_base_port,
                )
            else:
                # SGLang uses Dynamo's ZmqKvEventListener (JSON protocol).
                for idx, wid in enumerate(discovered_ids):
                    endpoint = f"tcp://127.0.0.1:{kv_event_base_port + idx}"
                    self.indexer.add_worker(wid, endpoint)
                self.indexer.start_background_drain(interval=0.25)
                logger.info(
                    "KvIndexer: %d SGLang workers registered, background drain started (base_port=%d)",
                    len(discovered_ids),
                    kv_event_base_port,
                )
        else:
            logger.info(
                "KvIndexer: KV event drain disabled (ENABLE_KV_EVENTS=%s, KV_EVENT_BASE_PORT=%s); "
                "using record_routing_decision() for radix tree updates",
                os.environ.get("ENABLE_KV_EVENTS", "unset"),
                os.environ.get("KV_EVENT_BASE_PORT", "unset"),
            )

        self._initialize_bandits()
        self._initialize_contextual()
        logger.info("WorkloadAwareRouter initialized with %d backend worker(s)",
                    len(list(self.engine_client.instance_ids())))

        if self.router_type == "kv_thompson":
            qpw = self.kv_thompson_queue_pw if self.kv_thompson_queue_pw is not None else self.queue_penalty_weight
            logger.info(
                "kv_thompson config: ts_weight=%.4f idle_boost=%.4f temperature=%.4f "
                "queue_penalty_weight=%.4f cold_start_threshold=%.4f metrics_scrape_interval=%.3f",
                self.kv_ts_weight, self.idle_boost, self.kv_load_temp,
                qpw, self.kv_thompson_cold_start, self.metrics_scrape_interval,
            )
            features = []
            if self.kt_enable_lints:
                features.append(f"lints(w={self.kt_lints_weight:.2f})")
            if self.kt_enable_affinity:
                features.append(f"affinity(base={self.kt_affinity_base:.2f} reuse={self.kt_affinity_reuse_weight:.2f})")
            if self.kt_enable_switching_cost:
                features.append(f"switch_cost(base={self.kt_switch_base:.2f} reuse={self.kt_switch_reuse:.2f})")
            if self.kt_enable_full_load:
                features.append("full_load(gpu+queue+outstanding+coupling)")
            if self.kt_enable_adaptive_temp:
                features.append(f"adaptive_temp(base={self.kt_adaptive_temp_base:.2f})")
            if self.kt_enable_adaptive_explore:
                features.append("adaptive_explore")
            if self.kt_enable_sticky_floor:
                features.append(f"sticky_floor({self.kt_sticky_load_floor:.3f})")
            if features:
                logger.info("kv_thompson features ON: %s", " | ".join(features))
            else:
                logger.info("kv_thompson features: all optional features OFF (base mode)")

    @safe_update("_init_lock")
    def _initialize_bandits(self):
        for wid in self.engine_client.instance_ids():
            wid = int(wid)
            self.beta_learner.add_worker(wid)
            if self._metrics.get("beta_alpha"):
                self._metrics["beta_alpha"].labels(worker_id=str(wid)).set(1.0)
            if self._metrics.get("beta_beta"):
                self._metrics["beta_beta"].labels(worker_id=str(wid)).set(1.0)

    @safe_update("_init_lock")
    def _initialize_contextual(self):
        for wid in self.engine_client.instance_ids():
            wid = int(wid)
            self.lints_learner.add_worker(wid)

    # --------------------- prefix state --------------------- #
    @safe_update("_prefix_lock")
    def _get_prefix(self, pid: str) -> tuple[int | None, int]:
        info = self.prefix_cache_state.get(pid)
        if info:
            return info.get("worker"), int(info.get("reuse_remaining") or 0)
        return None, 0

    @safe_update("_prefix_lock")
    def _set_prefix(
        self,
        pid: str,
        wid: int,
        reuse_remaining: int,
        decode_cost: float,
        prefill_cost: float,
        iat_factor: float,
        consecutive: int = 1,
    ):
        """Record/refresh prefix assignment."""
        if reuse_remaining <= 0:
            self.prefix_cache_state.pop(pid, None)
            self.prefix_meta.pop(pid, None)
        else:
            self.prefix_cache_state[pid] = {
                "worker": wid,
                "reuse_remaining": max(0, int(reuse_remaining)),
                "consecutive": int(consecutive),
            }
            self.prefix_meta[pid] = {
                "decode_cost": float(decode_cost),
                "prefill_cost": float(max(prefill_cost, 0.0)),
                "iat_factor": float(iat_factor),
            }

        # Update prefix state size metric
        if self._metrics.get("prefix_state_size"):
            self._metrics["prefix_state_size"].set(len(self.prefix_cache_state))

    def _worker_outstanding(self, wid: int) -> tuple[int, float]:
        """Returns (reuse_total, work_total) for a worker."""
        reuse_total = 0
        work_total = 0.0
        for pid, info in self.prefix_cache_state.items():
            if info.get("worker") != wid:
                continue
            r = int(info.get("reuse_remaining") or 0)
            reuse_total += r
            meta = self.prefix_meta.get(pid)
            if meta:
                work_total += float(r) * (float(meta.get("decode_cost", 2.0)) +
                                          float(meta.get("prefill_cost", 0.0))) * float(meta.get("iat_factor", 1.0))
        return reuse_total, work_total

    # Backend-agnostic metric line prefixes.
    # Each canonical metric maps to the exact line prefix(es) for SGLang and vLLM.
    # Using startswith() avoids substring collisions (e.g. pending_prealloc_token_usage).
    _METRIC_PREFIXES: dict[str, list[str]] = {
        "locked_kv_cache": [
            "sglang:token_usage{",  # SGLang: locked (non-evictable) KV cache fraction (0-1)
            "vllm:kv_cache_usage_perc{",  # vLLM: same semantic — only ref_cnt>0 blocks
        ],
        "queue_depth": [
            "sglang:num_queue_reqs{",  # SGLang: scheduler queue depth
            "vllm:num_requests_waiting{",  # vLLM: same semantic
        ],
        "num_used_tokens": [
            "sglang:num_used_tokens{",  # SGLang: locked tokens (total - available - evictable)
        ],
        "max_total_num_tokens": [
            "sglang:max_total_num_tokens{",  # SGLang: total KV cache capacity in tokens
        ],
        "active_seq_tokens": [
            "sglang:decode_sum_seq_lens{",  # SGLang: sum of sequence lengths of running batch
        ],
        "num_running_reqs": [
            "sglang:num_running_reqs{",  # SGLang: running batch size
            "vllm:num_requests_running{",  # vLLM: same semantic
        ],
    }

    # ---- cached metrics scraper (non-blocking) ---- #

    def _start_metrics_scraper(self, worker_ids: list[int], interval: float = 1.0) -> None:
        """Start a background thread that periodically scrapes worker metrics.

        The scrape runs in a daemon thread to avoid blocking the asyncio event
        loop.  Results are cached in ``_scraped_metrics`` and read lock-free
        by ``_build_internal_metrics`` on every routing decision.
        """
        if hasattr(self, "_scraper_running") and self._scraper_running:
            return

        self._scraped_metrics: dict[int, dict[str, float]] = {}
        self._scraper_running = True
        self._scraper_worker_ids = sorted(worker_ids)
        self._scraper_base_port = int(os.environ.get("WORKER_METRICS_PORT", "0"))

        def _scrape_loop() -> None:
            import urllib.request
            while self._scraper_running:
                for idx, wid in enumerate(self._scraper_worker_ids):
                    if self._scraper_base_port <= 0:
                        break
                    port = self._scraper_base_port + idx
                    scraped: dict[str, float] = {
                        "locked_kv_cache": 0.0,
                        "queue_depth": 0.0,
                        "num_used_tokens": 0.0,
                        "max_total_num_tokens": 0.0,
                        "active_seq_tokens": 0.0,
                        "num_running_reqs": 0.0,
                    }
                    try:
                        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1.0)
                        body = resp.read().decode("utf-8", errors="replace")
                        for line in body.splitlines():
                            if line.startswith("#"):
                                continue
                            for key, prefixes in self._METRIC_PREFIXES.items():
                                for prefix in prefixes:
                                    if line.startswith(prefix):
                                        scraped[key] = float(line.rsplit(" ", 1)[-1])
                                        break
                    except Exception:
                        pass
                    self._scraped_metrics[wid] = scraped
                time.sleep(interval)

        t = threading.Thread(target=_scrape_loop, daemon=True, name="metrics-scraper")
        t.start()
        logger.info("Started background metrics scraper (interval=%.1fs, workers=%d)", interval, len(worker_ids))

    def _build_internal_metrics(self, worker_ids: list[int]) -> dict[str, Any]:
        """Build a metrics dict from cached scrapes + instant pending counts.

        The worker metrics are scraped in a background thread (no event loop
        blocking).  Pending-decision counts provide an instant supplement that
        reacts within the same function call.
        """
        # Count in-flight (pending) decisions per worker.
        raw_counts = self.pending_decisions.per_worker_counts()
        pending_per_worker: dict[int, int] = {wid: raw_counts.get(wid, 0) for wid in worker_ids}

        sorted_ids = sorted(worker_ids)
        endpoints = []
        for wid in sorted_ids:
            pending = float(pending_per_worker.get(wid, 0))
            cached = getattr(self, "_scraped_metrics", {}).get(wid)

            if cached:
                locked_kv = cached.get("locked_kv_cache", 0.0)
                queue_depth = cached.get("queue_depth", 0.0)
                num_used_tokens = cached.get("num_used_tokens", 0.0)
                max_total_tokens = cached.get("max_total_num_tokens", 0.0)
                active_seq_tokens = cached.get("active_seq_tokens", 0.0)
                num_running_reqs = cached.get("num_running_reqs", 0.0)
            else:
                locked_kv = min(1.0, pending / 20.0)
                queue_depth = pending
                num_used_tokens = 0.0
                max_total_tokens = 0.0
                active_seq_tokens = 0.0
                num_running_reqs = pending

            effective_queue = max(queue_depth, pending)
            effective_running = max(num_running_reqs, pending)

            endpoints.append({
                "worker_id": wid,
                "num_requests_waiting": effective_queue,
                "locked_kv_cache_perc": locked_kv,
                "num_used_tokens": num_used_tokens,
                "max_total_num_tokens": max_total_tokens,
                "active_seq_tokens": active_seq_tokens,
                "num_running_reqs": effective_running,
            })

        return {"endpoints": endpoints}

    # --------------------- bandits (delegated to learners) --------------------- #
    def _linTS_sample(self, wid: int, x: np.ndarray) -> float:
        return self.lints_learner.sample(wid, x)

    def _update_contextual(self, wid: int, x: np.ndarray, reward: float):
        self.lints_learner.update(wid, x, reward)

    def _ts_sample(self, worker_id: int) -> float:
        return self.beta_learner.sample(worker_id)

    def _update_bandit(self, worker_id: int, reward: float):
        new_alpha, new_beta = self.beta_learner.update(worker_id, reward)
        if self._metrics.get("beta_alpha"):
            self._metrics["beta_alpha"].labels(worker_id=str(worker_id)).set(new_alpha)
        if self._metrics.get("beta_beta"):
            self._metrics["beta_beta"].labels(worker_id=str(worker_id)).set(new_beta)

    # --------------------- features / scores --------------------- #
    def _prefill_cost_for_worker(self, tokens: list[int], overlap: float) -> float:
        isl = max(0, len(tokens))
        frac = min(max(float(overlap), 0.0), 1.0)
        uncached = max(0.0, float(isl) * (1.0 - frac))
        return (uncached / self.prefill_token_scale) * self.prefill_weight

    @staticmethod
    def _prefill_bin(prefill_cost: float) -> str:
        if prefill_cost < 0.25:
            return "LOW"
        if prefill_cost < 0.75:
            return "MEDIUM"
        return "HIGH"

    def _feature_vector(
        self,
        wid: int,
        metrics: dict[str, Any] | None,
        scores: "OverlapScores",
        last_w: int | None,
        reuse_after: int,
        decode_cost: float,
        prefill_cost: float,
        iat_factor: float,
    ) -> np.ndarray:
        locked_kv = 0.0
        queue = 0.0
        if metrics and isinstance(metrics, dict) and "endpoints" in metrics:
            for ep in metrics["endpoints"]:
                if ep.get("worker_id") == wid:
                    locked_kv = float(ep.get("locked_kv_cache_perc", 0.0))
                    queue = float(ep.get("num_requests_waiting", 0.0))
                    break
        inv_load = 1.0 / (1.0 + self.gpu_penalty_weight * max(0.0, locked_kv) + self.queue_penalty_weight * max(0.0, queue))

        overlap = float(scores.scores.get(wid, 0.0))
        affinity = 1.0 if (last_w is not None and wid == last_w) else 0.0
        _, work_out = self._worker_outstanding(wid)

        decode_norm = decode_cost / 3.0
        prefill_norm = math.tanh(prefill_cost)
        iat_norm = iat_factor / 1.5
        outstanding_norm = math.tanh(0.1 * work_out)
        reuse_norm = math.tanh(0.25 * float(max(reuse_after, 0)))

        return np.array([
            1.0,
            inv_load,
            overlap,
            affinity,
            outstanding_norm,
            decode_norm,
            prefill_norm,
            iat_norm,
            reuse_norm,
        ],
                        dtype=np.float64)

    def _load_score(self, wid: int, metrics: dict[str, Any] | None, job_cost_total: float) -> float:
        locked_kv = 0.0
        queue = 0.0
        if metrics and isinstance(metrics, dict) and "endpoints" in metrics:
            for ep in metrics["endpoints"]:
                if ep.get("worker_id") == wid:
                    locked_kv = float(ep.get("locked_kv_cache_perc", 0.0))
                    queue = float(ep.get("num_requests_waiting", 0.0))
                    break
        _, work_out = self._worker_outstanding(wid)
        penalty = (self.gpu_penalty_weight * locked_kv + self.queue_penalty_weight * queue +
                   self.outstanding_work_weight * max(0.0, work_out) +
                   self.job_gpu_coupling_weight * job_cost_total * locked_kv +
                   self.job_queue_coupling_weight * job_cost_total * queue)
        return 1.0 / (1.0 + max(0.0, penalty))

    def _softmax(self, scores: list[float], temp: float) -> list[float]:
        t = float(min(max(temp, self.temp_min), self.temp_max))
        m = float(np.max(scores))
        exps = np.exp((np.array(scores) - m) / max(1e-6, t))
        s = float(np.sum(exps))
        if s <= 0.0 or not np.isfinite(s):
            return [1.0 / len(scores)] * len(scores)
        return list((exps / s).astype(float))

    # --------------------- selection --------------------- #
    def _select_worker(
        self,
        worker_ids,
        req: RouterRequest,
        metrics: dict[str, Any] | None,
        scores: OverlapScores,
    ) -> tuple[int, dict[str, float], dict[int, dict[str, float]], list[float], list[float]]:
        last_w, _ = self._get_prefix(req.prefix_id)

        reuse_after = max(int(req.reuse_budget), 0)
        decode_cost = self._decode_cost(req.expected_osl)
        iat_factor = self._iat_factor(req.interarrival)

        # ---- kv_only: route purely on KV overlap scores ---- #
        if self.router_type == "kv_only":
            worker_list = [int(w) for w in worker_ids]
            raw_scores: list[float] = []
            per_worker_ctx: dict[int, dict[str, float]] = {}
            all_overlaps: dict[int, float] = {}
            for wid in worker_list:
                overlap = float(scores.scores.get(wid, 0.0))
                prefill_cost = self._prefill_cost_for_worker(req.tokens, overlap)
                raw_scores.append(overlap)
                all_overlaps[wid] = overlap
                per_worker_ctx[wid] = {
                    "decode_cost": decode_cost,
                    "prefill_cost": prefill_cost,
                    "iat_factor": iat_factor,
                    "overlap": overlap,
                    "reuse_after": float(reuse_after),
                    "load_mod": 1.0,
                }

            best = max(raw_scores) if raw_scores else 0.0

            # Ignore trivially small overlaps (shared BOS/template noise).
            # If best overlap is below ~5% the match is not meaningful domain
            # affinity -- use round-robin to guarantee every worker gets seeded
            # before any worker gets a second cold-start assignment.
            min_meaningful_overlap = 0.05
            if best < min_meaningful_overlap:
                candidates = list(range(len(worker_list)))
                idx = self._cold_start_rr % len(worker_list)
                self._cold_start_rr += 1
            else:
                candidates = [i for i, s in enumerate(raw_scores) if s >= best - 1e-6]
                idx = random.choice(candidates)
            chosen = int(worker_list[idx])

            probs = [0.0] * len(worker_list)
            for c in candidates:
                probs[c] = 1.0 / len(candidates)

            overlap_str = " ".join(f"w{wid}={v:.3f}" for wid, v in sorted(all_overlaps.items()))
            logger.info(
                "kv_only: prefix=%s chosen=%s candidates=%d/%d best=%.4f overlaps=[%s]",
                req.prefix_id, chosen, len(candidates), len(worker_list), best, overlap_str,
            )

            return chosen, per_worker_ctx[chosen], per_worker_ctx, raw_scores, probs

        # ---- kv_load_balanced: kv_load formula + idle_boost + softmax ---- #
        # Same as kv_load but with:
        #   - idle_boost: idle workers get a minimum overlap credit
        #   - softmax with temperature instead of deterministic argmin
        if self.router_type == "kv_load_balanced":
            worker_list = [int(w) for w in worker_ids]
            raw_scores: list[float] = []
            per_worker_ctx: dict[int, dict[str, float]] = {}
            all_overlaps: dict[int, float] = {}
            request_blocks = math.ceil(len(req.tokens) / self.block_size)

            for wid in worker_list:
                overlap = float(scores.scores.get(wid, 0.0))
                prefill_cost = self._prefill_cost_for_worker(req.tokens, overlap)

                overlap_blocks = scores.raw_block_counts.get(wid, 0)
                boosted_blocks = max(overlap_blocks, int(self.idle_boost * scores.total_blocks)) if scores.total_blocks > 0 else overlap_blocks
                prefill_tokens = max(0, len(req.tokens) - boosted_blocks * self.block_size)
                potential_prefill_blocks = prefill_tokens / self.block_size

                routed = float(self._kv_load_routed_count(wid))
                decode_blocks = routed * request_blocks

                logit = self.kv_load_overlap_weight * potential_prefill_blocks + decode_blocks
                score = -logit

                raw_scores.append(score)
                all_overlaps[wid] = max(overlap, self.idle_boost)
                per_worker_ctx[wid] = {
                    "decode_cost": decode_cost,
                    "prefill_cost": prefill_cost,
                    "iat_factor": iat_factor,
                    "overlap": overlap,
                    "reuse_after": float(reuse_after),
                    "load_mod": 1.0 / (1.0 + routed),
                }

            probs = self._softmax(raw_scores, self.kv_load_temp)
            r = random.random()
            cum = 0.0
            idx = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    idx = i
                    break
            chosen = int(worker_list[idx])

            self._kv_load_track(chosen)
            detail = " ".join(
                f"w{wid}=[ov={all_overlaps[wid]:.3f} sc={raw_scores[i]:.3f}]"
                for i, wid in enumerate(worker_list)
            )
            logger.info(
                "kv_load_balanced: prefix=%s chosen=%s workers=[%s]",
                req.prefix_id, chosen, detail,
            )

            return chosen, per_worker_ctx[chosen], per_worker_ctx, raw_scores, probs

        # ---- kv_thompson: modular scoring with togglable features ---- #
        if self.router_type == "kv_thompson":
            worker_list = [int(w) for w in worker_ids]
            raw_scores: list[float] = []
            per_worker_ctx: dict[int, dict[str, float]] = {}
            all_overlaps: dict[int, float] = {}

            qpw = self.kv_thompson_queue_pw if self.kv_thompson_queue_pw is not None else self.queue_penalty_weight

            for wid in worker_list:
                overlap = float(scores.scores.get(wid, 0.0))
                prefill_cost = self._prefill_cost_for_worker(req.tokens, overlap)
                job_cost_total = decode_cost + prefill_cost

                effective_overlap = max(overlap, self.idle_boost)

                q = 0.0  # raw queue depth (populated in non-full-load path)
                if self.kt_enable_full_load:
                    load_mod = self._load_score(wid, metrics, job_cost_total=job_cost_total)
                else:
                    queue = 0.0
                    if metrics and isinstance(metrics, dict) and "endpoints" in metrics:
                        for ep in metrics["endpoints"]:
                            if ep.get("worker_id") == wid:
                                queue = float(ep.get("num_requests_waiting", 0.0))
                                break
                    # Exponential quadratic penalty: gentle at low queue, aggressive at high.
                    # load_mod = exp(-qpw * queue² / 25)
                    # At queue=1: ~exp(-qpw/25) ≈ 0.90 (barely noticeable)
                    # At queue=5: exp(-qpw) (the "knee" — qpw directly controls severity here)
                    # At queue=10: exp(-4*qpw) (effectively zero)
                    q = max(0.0, queue)
                    load_mod = math.exp(-qpw * q * q / 25.0)

                if self.kt_enable_sticky_floor and last_w == wid and reuse_after > 0:
                    load_mod = max(load_mod, self.kt_sticky_load_floor)

                # Global minimum floor: prevents load_mod from collapsing to
                # zero under high queue depth, which would erase the KV overlap
                # signal entirely and leave only LinTS to route.
                if self.kt_load_mod_floor > 0.0:
                    load_mod = max(load_mod, self.kt_load_mod_floor)

                score_base = effective_overlap * load_mod
                score = score_base

                if self.kt_enable_adaptive_explore:
                    ts_w_eff = self.kv_ts_weight / (1.0 + float(reuse_after) * iat_factor)
                else:
                    ts_w_eff = self.kv_ts_weight
                score += ts_w_eff * self._ts_sample(wid)

                score_lints = 0.0
                if self.kt_enable_lints:
                    x = self._feature_vector(
                        wid=wid, metrics=metrics, scores=scores, last_w=last_w,
                        reuse_after=reuse_after, decode_cost=decode_cost,
                        prefill_cost=prefill_cost, iat_factor=iat_factor,
                    )
                    raw_lints = self._linTS_sample(wid, x)
                    if self.kt_lints_weight < 0:
                        # Negative weight enables tanh normalization: bounds
                        # LinTS output to [-1, 1] then scales by |weight|.
                        # This keeps LinTS as a tiebreaker rather than the
                        # dominant signal.
                        score_lints = abs(self.kt_lints_weight) * math.tanh(raw_lints)
                    else:
                        score_lints = self.kt_lints_weight * raw_lints
                    score += score_lints

                score_affinity = 0.0
                if self.kt_enable_affinity and last_w == wid and reuse_after > 0:
                    # Scale affinity bonus by load_mod so that stickiness shrinks
                    # as the worker's queue grows.  This prevents the lock-in
                    # cascade where a flat affinity bonus overrides the queue
                    # penalty and traps sessions on an overloaded worker.
                    # At load_mod=1 (idle):   full affinity bonus preserved
                    # At load_mod=0.1 (floor, deep queue): bonus is 10% of normal
                    # → high-overlap idle alternatives can now win the decision.
                    score_affinity = (self.kt_affinity_base
                                      + self.kt_affinity_reuse_weight * float(reuse_after)) * (0.5 + 0.5 * overlap) * load_mod
                    score += score_affinity

                score_switch = 0.0
                if self.kt_enable_switching_cost and last_w is not None and wid != last_w and reuse_after > 0:
                    score_switch = -(self.kt_switch_base
                                     + self.kt_switch_reuse * float(reuse_after))
                    score += score_switch

                if np.isnan(score) or np.isinf(score):
                    score = -1e9

                raw_scores.append(float(score))
                all_overlaps[wid] = overlap
                per_worker_ctx[wid] = {
                    "decode_cost": decode_cost,
                    "prefill_cost": prefill_cost,
                    "iat_factor": iat_factor,
                    "overlap": overlap,
                    "reuse_after": float(reuse_after),
                    "load_mod": load_mod,
                    "queue": q,
                    "score_base": score_base,
                    "score_lints": score_lints,
                    "score_affinity": score_affinity,
                    "score_switch": score_switch,
                }

            best = max(all_overlaps.values()) if all_overlaps else 0.0
            if best < self.kv_thompson_cold_start:
                idx = self._cold_start_rr % len(worker_list)
                self._cold_start_rr += 1
                chosen = int(worker_list[idx])
                probs = [1.0 / len(worker_list)] * len(worker_list)
                logger.info(
                    "kv_thompson: COLD_START prefix=%s chosen=%s best_ov=%.4f "
                    "threshold=%.4f rr_idx=%d/%d",
                    req.prefix_id, chosen, best,
                    self.kv_thompson_cold_start, idx, len(worker_list),
                )
            else:
                if self.kt_enable_adaptive_temp:
                    temp = self.kt_adaptive_temp_base / (1.0 + float(reuse_after) * iat_factor)
                    temp = min(max(temp, self.temp_min), self.temp_max)
                else:
                    temp = self.kv_load_temp
                probs = self._softmax(raw_scores, temp)
                r = random.random()
                cum = 0.0
                idx = 0
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        idx = i
                        break
                chosen = int(worker_list[idx])

            detail = " ".join(
                f"w{wid}=[ov={all_overlaps[wid]:.3f} q={int(per_worker_ctx[wid]['queue'])} "
                f"ld={per_worker_ctx[wid]['load_mod']:.3f} "
                f"sc={raw_scores[i]:.3f} p={probs[i]:.3f}]"
                for i, wid in enumerate(worker_list)
            )
            logger.info(
                "kv_thompson: prefix=%s chosen=%s best_ov=%.4f "
                "params=[ts_w=%.4f idle=%.4f qpw=%.4f temp=%.4f cold=%.4f] "
                "features=[lints=%s affinity=%s switch=%s full_load=%s adapt_t=%s adapt_e=%s sticky=%s] "
                "workers=[%s]",
                req.prefix_id, chosen, best,
                self.kv_ts_weight, self.idle_boost, qpw,
                self.kv_load_temp, self.kv_thompson_cold_start,
                self.kt_enable_lints, self.kt_enable_affinity,
                self.kt_enable_switching_cost, self.kt_enable_full_load,
                self.kt_enable_adaptive_temp, self.kt_enable_adaptive_explore,
                self.kt_enable_sticky_floor,
                detail,
            )
            # Per-component breakdown for the chosen worker to aid debugging.
            if chosen in per_worker_ctx:
                ctx_c = per_worker_ctx[chosen]
                logger.info(
                    "kv_thompson chosen_breakdown: prefix=%s worker=%s q=%d "
                    "base=%+.3f lints=%+.3f affinity=%+.3f switch=%+.3f total=%.3f",
                    req.prefix_id, chosen, int(ctx_c["queue"]),
                    ctx_c["score_base"], ctx_c["score_lints"],
                    ctx_c["score_affinity"], ctx_c["score_switch"],
                    raw_scores[worker_list.index(chosen)],
                )

            # ---- KV opportunity-cost analysis ----
            # For every routing decision, find the "best idle" worker (q=0 or
            # lowest queue) and compare its KV overlap with the chosen worker.
            # This shows the trade-off: how much KV coverage did we gain/lose
            # by routing to a busier worker instead of the most available one?
            chosen_q    = int(per_worker_ctx[chosen]["queue"])
            chosen_ov   = all_overlaps[chosen]

            # Find best-idle: among q=0 workers, pick highest overlap.
            # Fall back to lowest-queue worker if none are fully idle.
            idle_workers = [(wid, all_overlaps[wid], int(per_worker_ctx[wid]["queue"]))
                            for wid in worker_list]
            min_q = min(q for _, _, q in idle_workers)
            min_q_workers = [(wid, ov) for wid, ov, q in idle_workers if q == min_q]
            best_idle_wid, best_idle_ov = max(min_q_workers, key=lambda x: x[1])

            kv_delta = chosen_ov - best_idle_ov  # + means we're on a better KV worker
            kv_available_on_idle = best_idle_ov  # how much we'd get if we routed to idle

            # Estimate uncached tokens for chosen vs idle paths
            isl = len(req.tokens)
            uncached_chosen = max(0.0, isl * (1.0 - chosen_ov))
            uncached_idle   = max(0.0, isl * (1.0 - best_idle_ov))
            extra_cold_tokens = uncached_idle - uncached_chosen  # tokens we save by staying sticky

            if chosen_q > 0 and min_q == 0:
                # Routed to a queued worker while idle capacity existed
                logger.info(
                    "kv_opportunity: prefix=%s QUEUED_PREFERRED: chosen=w%s q=%d ov=%.3f "
                    "vs idle=w%s q=0 ov=%.3f  kv_delta=%+.3f  "
                    "extra_cold_tokens=%.0f (ISL=%d tokens — KV gained by staying sticky)",
                    req.prefix_id, str(chosen)[-5:], chosen_q, chosen_ov,
                    str(best_idle_wid)[-5:], best_idle_ov,
                    kv_delta, extra_cold_tokens, isl,
                )
                if self._metrics.get("routed_to_queued"):
                    self._metrics["routed_to_queued"].inc()
            elif chosen == best_idle_wid:
                # Chose the lowest-queue worker AND it had good KV coverage
                logger.debug(
                    "kv_opportunity: prefix=%s IDLE_MATCH: chosen=w%s q=%d ov=%.3f (optimal)",
                    req.prefix_id, str(chosen)[-5:], chosen_q, chosen_ov,
                )
            else:
                # Both workers have queue — picked higher-overlap one
                logger.debug(
                    "kv_opportunity: prefix=%s ALL_BUSY: chosen=w%s q=%d ov=%.3f "
                    "vs best_low_q=w%s q=%d ov=%.3f  kv_delta=%+.3f",
                    req.prefix_id, str(chosen)[-5:], chosen_q, chosen_ov,
                    str(best_idle_wid)[-5:], min_q, best_idle_ov, kv_delta,
                )

            # Prometheus metrics (emitted on every decision)
            if self._metrics.get("kv_vs_idle_delta"):
                self._metrics["kv_vs_idle_delta"].observe(kv_delta)
            if self._metrics.get("idle_kv_miss"):
                self._metrics["idle_kv_miss"].observe(best_idle_ov)

            return chosen, per_worker_ctx[chosen], per_worker_ctx, raw_scores, probs

        temp = self.temp_base / (1.0 + float(reuse_after) * iat_factor)
        temp = min(max(temp, self.temp_min), self.temp_max)

        raw_scores: list[float] = []
        worker_list: list[int] = [int(w) for w in worker_ids]
        per_worker_ctx: dict[int, dict[str, float]] = {}
        load_mods: list[float] = []
        overlaps: list[float] = []

        for wid in worker_list:
            overlap = float(scores.scores.get(wid, 0.0))
            prefill_cost = self._prefill_cost_for_worker(req.tokens, overlap)
            job_cost_total = decode_cost + prefill_cost

            x = self._feature_vector(
                wid=wid,
                metrics=metrics,
                scores=scores,
                last_w=last_w,
                reuse_after=reuse_after,
                decode_cost=decode_cost,
                prefill_cost=prefill_cost,
                iat_factor=iat_factor,
            )

            val = self._linTS_sample(wid, x)
            explore_w = self.base_ts_weight / (1.0 + float(reuse_after) * iat_factor)
            val += explore_w * self._ts_sample(wid)

            if last_w == wid and (reuse_after > 0):
                val += (self.affinity_base + self.affinity_reuse_weight * float(reuse_after)) * (0.5 + 0.5 * overlap)

            if last_w is not None and wid != last_w and (reuse_after > 0):
                val -= (self.switch_cost_base + self.switch_cost_reuse * float(reuse_after))

            load_mod = self._load_score(wid, metrics, job_cost_total=job_cost_total)
            if last_w == wid and reuse_after > 0:
                load_mod = max(load_mod, self.sticky_load_floor)
            val *= load_mod

            if np.isnan(val) or np.isinf(val):
                val = -1e9

            raw_scores.append(float(val))
            load_mods.append(float(load_mod))
            overlaps.append(float(overlap))
            per_worker_ctx[wid] = {
                "decode_cost": decode_cost,
                "prefill_cost": prefill_cost,
                "iat_factor": iat_factor,
                "overlap": overlap,
                "reuse_after": float(reuse_after),
                "load_mod": load_mod,
            }

        probs = self._softmax(raw_scores, temp)
        r = random.random()
        cum = 0.0
        idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        chosen = int(worker_list[idx])

        return chosen, per_worker_ctx[chosen], per_worker_ctx, raw_scores, probs

    # --------------------- latency baselines & reward (delegated) --------------------- #
    def _get_latency_baseline(self, wid: int, osl: str, prefill_bin: str, per_tok: bool, fallback: float) -> float:
        return self.latency_tracker.get_baseline(wid, osl, prefill_bin, per_tok, fallback)

    def _update_latency_baselines(self, wid: int, osl: str, prefill_bin: str, metric: float, per_tok: bool) -> float:
        return self.latency_tracker.update_baselines(wid, osl, prefill_bin, metric, per_tok)

    @staticmethod
    def _latency_metric(latency_ms: float, tokens_out: int | None) -> tuple[float, bool]:
        return LatencyTracker.latency_metric(latency_ms, tokens_out)

    @staticmethod
    def _metric_to_reward(metric: float, baseline: float, success: bool) -> float:
        return LatencyTracker.compute_reward(metric, baseline, success)

    # --------------------- timeout sweep --------------------- #
    def _sweep_pending(self, now: float):
        expired = self.pending_decisions.sweep(now)
        if not expired:
            return

        if self._metrics.get("pending_decisions"):
            self._metrics["pending_decisions"].set(self.pending_decisions.count())

        for did, rec in expired:
            wid = int(rec["wid"])
            x = rec["x"]
            reward = float(self.timeout_reward)
            self._update_bandit(wid, reward)
            self._update_contextual(wid, x, reward)

            if self._metrics.get("timeout_penalties"):
                self._metrics["timeout_penalties"].inc()

            self._emit_trace(
                "timeout",
                {
                    "decision_id": did,
                    "wid": wid,
                    "reward": reward,
                    "age": self.feedback_timeout_seconds,
                    "prefix_id": rec.get("prefix_id"),
                    "osl": rec.get("osl"),
                    "prefill_bin": rec.get("prefill_bin"),
                })
            logger.warning("Timeout feedback: wid=%s decision=%s reward=%.3f", wid, did, reward)

    # --------------------- main endpoint: find_worker --------------------- #
    async def generate(self, request: dict):
        req = RouterRequest(**request)

        worker_ids = [int(w) for w in self.engine_client.instance_ids()]
        if not worker_ids:
            yield RouterResponse(worker_id=-1, prefix_hit_rate=0.0).model_dump()
            return

        now = time.time()
        self._sweep_pending(now)

        # Track tokens per request
        if self._metrics.get("tokens_per_request"):
            self._metrics["tokens_per_request"].observe(len(req.tokens))
        if self._metrics.get("reuse_budget"):
            self._metrics["reuse_budget"].observe(req.reuse_budget)

        metrics = self._build_internal_metrics(worker_ids)
        if self.router_type == "kv_load":
            scores: OverlapScores = await self.indexer.find_matches_for_request(req.tokens, 0)
            chosen, overlap_chosen = self._select_worker_kv_load(worker_ids, req.tokens, metrics, scores)
            self.indexer.record_routing_decision(chosen, req.tokens)
            self._kv_load_track(chosen)
            yield RouterResponse(worker_id=chosen, prefix_hit_rate=overlap_chosen).model_dump()
            return

        scores: OverlapScores = await self.indexer.find_matches_for_request(req.tokens, 0)
        chosen, chosen_ctx, all_ctx, raw_scores, probs = self._select_worker(worker_ids, req, metrics, scores)

        # Record this decision so the local radix tree tracks which blocks
        # are cached on which worker (same strategy as Dynamo's --no-kv-events mode).
        self.indexer.record_routing_decision(chosen, req.tokens)

        last_w, _ = self._get_prefix(req.prefix_id)

        # Consecutive same-worker counter: how many calls in a row this session
        # has been routed to the same worker.  Used to detect affinity lock-in.
        prev_consecutive = int(self.prefix_cache_state.get(req.prefix_id, {}).get("consecutive", 0))
        consecutive = prev_consecutive + 1 if last_w == chosen else 1

        decode_cost = self._decode_cost(req.expected_osl)
        overlap_chosen = float(scores.scores.get(chosen, 0.0))
        prefill_cost_chosen = self._prefill_cost_for_worker(req.tokens, overlap_chosen)
        iat_factor = self._iat_factor(req.interarrival)

        # Update prefix state
        self._set_prefix(
            req.prefix_id,
            chosen,
            reuse_remaining=max(int(req.reuse_budget), 0),
            decode_cost=decode_cost,
            prefill_cost=prefill_cost_chosen,
            iat_factor=iat_factor,
            consecutive=consecutive,
        )

        # Build feature x for chosen & store pending decision
        x = self._feature_vector(
            wid=chosen,
            metrics=metrics,
            scores=scores,
            last_w=last_w,
            reuse_after=max(int(req.reuse_budget), 0),
            decode_cost=decode_cost,
            prefill_cost=prefill_cost_chosen,
            iat_factor=iat_factor,
        )
        decision_id = uuid.uuid4().hex
        self.pending_decisions.add(decision_id, {
            "wid": int(chosen),
            "x": x,
            "osl": self._osl_bin(req.expected_osl),
            "prefill_bin": self._prefill_bin(prefill_cost_chosen),
            "start_ts": now,
            "prefix_id": req.prefix_id,
            "tokens_in": len(req.tokens),
            "reuse_after": int(req.reuse_budget),
            "overlap": overlap_chosen,
            "prefill_cost": float(prefill_cost_chosen),
            "decode_cost": float(decode_cost),
            "consecutive": consecutive,
        })
        if self._metrics.get("pending_decisions"):
            self._metrics["pending_decisions"].set(self.pending_decisions.count())

        # Update Prometheus metrics
        if self._metrics.get("decisions_total"):
            self._metrics["decisions_total"].labels(worker_id=str(chosen)).inc()
        if self._metrics.get("kv_overlap"):
            self._metrics["kv_overlap"].labels(worker_id=str(chosen)).set(overlap_chosen)
        if self._metrics.get("decisions_by_domain"):
            domain = req.prefix_id.rsplit("-", 1)[0] if req.prefix_id else "unknown"
            self._metrics["decisions_by_domain"].labels(worker_id=str(chosen), domain=domain).inc()

        # Track sticky vs switch decisions
        if last_w is not None:
            if chosen == last_w:
                if self._metrics.get("sticky_decisions"):
                    self._metrics["sticky_decisions"].inc()
            elif self._metrics.get("switch_decisions"):
                self._metrics["switch_decisions"].inc()

        # Decision trace
        if self.debug_traces:
            worker_list = [int(w) for w in worker_ids]
            details = {
                wid: {
                    "score": float(raw_scores[i]),
                    "prob": float(probs[i]),
                    **all_ctx[wid],
                }
                for i, wid in enumerate(worker_list)
            }
            self._emit_trace("decision",
                             {
                                 "decision_id": decision_id,
                                 "prefix_id": req.prefix_id,
                                 "chosen": int(chosen),
                                 "workers": details,
                             })

        logger.info(
            "Router picked worker=%s decision=%s prefix=%s (last=%s reuse_after=%s osl=%s "
            "prefill_cost=%.3f iat=%s overlap=%.3f consecutive=%d)",
            chosen,
            decision_id,
            req.prefix_id,
            last_w,
            req.reuse_budget,
            req.expected_osl,
            prefill_cost_chosen,
            req.interarrival,
            overlap_chosen,
            consecutive,
        )

        resp = RouterResponse(worker_id=chosen, prefix_hit_rate=overlap_chosen, decision_id=decision_id)
        yield resp.model_dump()
        return

    # --------------------- feedback endpoint --------------------- #
    async def feedback(self, request: dict):
        """Ex-post reward update from processor with observed latency."""
        try:
            fb = FeedbackRequest(**request)
        except Exception as e:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error=str(e))
            yield ack.model_dump()
            return

        decision = self.pending_decisions.pop(fb.decision_id)
        if self._metrics.get("pending_decisions"):
            self._metrics["pending_decisions"].set(self.pending_decisions.count())

        if not decision:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error="unknown_decision")
            yield ack.model_dump()
            return

        wid: int = int(decision["wid"])
        x: np.ndarray = decision["x"]
        osl: str = str(decision["osl"])
        prefill_bin: str = str(decision["prefill_bin"])
        elapsed_ms = (time.time() - decision.get("start_ts", time.time())) * 1000
        consecutive: int = int(decision.get("consecutive", 0))
        tokens_out = None if fb.tokens_out is None else int(fb.tokens_out)
        metric, per_tok = LatencyTracker.latency_metric(float(fb.latency_ms), tokens_out)

        # Baseline for reward: mode selects which EMA level is used.
        # "global" (default) uses a shared baseline so fast workers get
        # higher rewards than slow ones.  "hierarchical" uses per-worker
        # baselines (legacy — equalizes rewards).  "blended" mixes both.
        if self.reward_baseline_mode == "global":
            baseline_before = self.latency_tracker.get_global_baseline(per_tok, fallback=metric)
        elif self.reward_baseline_mode == "blended":
            global_bl = self.latency_tracker.get_global_baseline(per_tok, fallback=metric)
            worker_bl = self.latency_tracker.get_baseline(wid, osl, prefill_bin, per_tok, fallback=metric)
            baseline_before = 0.7 * global_bl + 0.3 * worker_bl
        else:
            baseline_before = self.latency_tracker.get_baseline(wid, osl, prefill_bin, per_tok, fallback=metric)
        reward = LatencyTracker.compute_reward(metric, baseline_before, bool(fb.success))

        # Update all three EMA levels (global, worker, bucket) regardless
        # of which mode was used for reward — keeps diagnostics available.
        if fb.success:
            baseline_after = self.latency_tracker.update_baselines(wid, osl, prefill_bin, metric, per_tok)
        else:
            baseline_after = baseline_before

        # Update bandits with ex-post reward
        self._update_bandit(wid, reward)
        self._update_contextual(wid, x, reward)

        # Update Prometheus metrics
        if self._metrics.get("feedback_latency"):
            self._metrics["feedback_latency"].labels(worker_id=str(wid)).observe(fb.latency_ms / 1000.0)
        if self._metrics.get("reward"):
            self._metrics["reward"].labels(worker_id=str(wid)).set(reward)

        self._emit_trace(
            "feedback",
            {
                "decision_id": fb.decision_id,
                "wid": wid,
                "latency_ms": float(fb.latency_ms),
                "tokens_out": tokens_out,
                "metric": metric,
                "per_tok": per_tok,
                "baseline_used": baseline_before,
                "baseline_after": baseline_after,
                "reward": reward,
                "success": bool(fb.success),
                "finish_reason": fb.finish_reason or "",
            })

        logger.info(
            "Feedback: wid=%s decision=%s metric=%.3f%s baseline=%.3f reward=%.3f success=%s "
            "elapsed_ms=%.0f consecutive=%d",
            wid,
            fb.decision_id,
            metric,
            " ms/tok" if per_tok else " ms",
            baseline_before,
            reward,
            fb.success,
            elapsed_ms,
            consecutive,
        )

        ack = FeedbackAck(ok=True, used_baseline=float(baseline_before), reward=float(reward), worker_id=wid)
        yield ack.model_dump()
        return

    # --------------------- helpers --------------------- #

    def _kv_load_track(self, worker_id: int) -> None:
        """Record that a request was just routed to *worker_id*."""
        with self._kv_load_routed_lock:
            self._kv_load_routed[worker_id] = self._kv_load_routed.get(worker_id, 0) + 1

    def _kv_load_routed_count(self, worker_id: int) -> int:
        """Return cumulative routes to *worker_id*."""
        with self._kv_load_routed_lock:
            return self._kv_load_routed.get(worker_id, 0)

    def _select_worker_kv_load(
        self,
        worker_ids: list[int],
        tokens: list[int],
        metrics: dict[str, Any] | None,
        scores: OverlapScores,
    ) -> tuple[int, float]:
        """Select worker replicating Dynamo's DefaultWorkerSelector formula.

        Dynamo native (per worker):
            logit = overlap_weight * prefill_blocks + decode_blocks

        We replicate this with:
            prefill_blocks = (ISL - overlap * block_size) / block_size
            decode_blocks  = routed_count * avg_request_blocks

        routed_count is the number of requests we've previously routed to
        this worker (cumulative, tracked locally).  This is a direct analogue
        of Dynamo's ActiveSequences which counts active blocks per worker
        based on its own routing decisions — not scraped from the engine.

        The product routed_count * avg_request_blocks puts the decode term
        in the same block-count units as prefill_blocks so the two terms are
        comparable, and the decode term grows without bound as requests pile
        up, eventually forcing the router to spread load even when one worker
        has a perfect cache hit.

        Lower logit is better.  Temperature 0 = deterministic (Dynamo default).
        """
        if not worker_ids:
            wid = int(random.choice(list(self.engine_client.instance_ids())))
            return wid, 0.0

        isl_tokens = len(tokens)
        if isl_tokens == 0:
            wid = int(random.choice(worker_ids))
            return wid, 0.0

        overlap_weight = self.kv_load_overlap_weight
        temperature = self.kv_load_temperature
        request_blocks = math.ceil(isl_tokens / self.block_size)

        worker_logits: dict[int, float] = {}
        worker_overlaps: dict[int, int] = {}

        for wid in worker_ids:
            overlap_blocks = scores.raw_block_counts.get(wid, 0)
            worker_overlaps[wid] = overlap_blocks

            prefill_tokens = max(0, isl_tokens - overlap_blocks * self.block_size)
            potential_prefill_blocks = prefill_tokens / self.block_size

            routed = float(self._kv_load_routed_count(wid))
            decode_blocks = routed * request_blocks

            logit = overlap_weight * potential_prefill_blocks + decode_blocks

            worker_logits[wid] = logit

            logger.info(
                "kv_load: worker=%s overlap=%d prefill=%.1f "
                "decode=%.1f (routed=%.0f * req_blocks=%d) logit=%.1f",
                wid, overlap_blocks, potential_prefill_blocks,
                decode_blocks, routed, request_blocks, logit,
            )

        # Select worker via negative-logit softmax (lower logit = better = higher probability)
        candidates = self._softmax_select_min_logit(worker_logits, temperature, scores.tree_sizes)

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = min(candidates, key=lambda w: scores.tree_sizes.get(w, 0))

        total_blocks = scores.total_blocks
        overlap_frac = float(worker_overlaps.get(chosen, 0)) / max(1, total_blocks)

        tree_size = scores.tree_sizes.get(chosen, 0)
        logger.info(
            "kv_load: selected worker=%s logit=%.3f overlap_blocks=%d/%d (%.1f%%) tree_size=%d",
            chosen, worker_logits[chosen], worker_overlaps.get(chosen, 0),
            total_blocks, overlap_frac * 100, tree_size,
        )

        return chosen, overlap_frac

    @staticmethod
    def _softmax_select_min_logit(
        logits: dict[int, float],
        temperature: float,
        tree_sizes: dict[int, int] | None = None,
    ) -> list[int]:
        """Softmax selection where lower logit = better.

        Matches Dynamo's ``softmax_sample`` in scheduler.rs:
        - temperature == 0 → return all keys with the minimum logit
        - temperature > 0  → negate logits, apply softmax, sample one
        """
        if not logits:
            return []

        keys = list(logits.keys())
        if len(keys) == 1:
            return keys

        if temperature <= 0.0:
            min_val = min(logits.values())
            return [k for k, v in logits.items() if abs(v - min_val) < 1e-9]

        neg_logits = [-logits[k] for k in keys]
        max_neg = max(neg_logits)
        exp_values = [math.exp((v - max_neg) / temperature) for v in neg_logits]
        sum_exp = sum(exp_values)
        probabilities = [v / sum_exp for v in exp_values]

        sample = random.random()
        cumsum = 0.0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if sample <= cumsum:
                return [keys[i]]

        return [keys[-1]]

    def _get_underloaded(self, metrics: dict[str, Any] | None):
        if not metrics or not metrics.get("endpoints"):
            wid = int(random.choice(list(self.engine_client.instance_ids())))
            return wid, 0.0
        loads = {ep.get("worker_id"): ep.get("locked_kv_cache_perc", 0.0) for ep in metrics["endpoints"]}
        min_val = min(loads.values())
        candidates = [wid for wid, v in loads.items() if v == min_val]
        return random.choice(candidates), min_val


# ---------------------- worker entry point ---------------------- #
def parse_args():
    """Parse minimal CLI arguments.

    The router uses a YAML config file for most parameters.
    Only frequently-tuned parameters have dedicated CLI flags.
    Use --override for any other parameter.

    See PARAMETERS.md for full documentation.
    """
    parser = argparse.ArgumentParser(
        description="Optimized Thompson Sampling Router with Prometheus Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python router.py

  # Use custom config file
  python router.py --config /path/to/config.yaml

  # Override specific values
  python router.py --config config.yaml --affinity-base 0.5 --temp-base 1.5

  # Override any config value
  python router.py --config config.yaml --override load_balancing.gpu_penalty_weight=2.0

See PARAMETERS.md for full parameter documentation.
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml in script directory)",
    )

    # Primary tuning knobs (explicit CLI flags)
    parser.add_argument(
        "--affinity-base",
        type=float,
        default=None,
        help="Primary stickiness control [0.0-1.0] (overrides config)",
    )
    parser.add_argument(
        "--temp-base",
        type=float,
        default=None,
        help="Primary exploration control [0.15-2.0] (overrides config)",
    )
    parser.add_argument(
        "--lints-v",
        type=float,
        default=None,
        help="LinTS exploration variance [0.0-1.0] (overrides config)",
    )

    # Generic override for any config value
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config value using dot notation (repeatable)",
    )

    return parser.parse_args()


# -------------------------- learner management HTTP server -------------- #

class RouterManagementServer:
    """Lightweight HTTP server for router learner state, config, and reset.

    Runs alongside the main NATS endpoints on a separate port (default 8085).
    """

    def __init__(self, router: WorkloadAwareRouter, config_path: str, port: int = 8085):
        self._router = router
        self._config_path = config_path
        self._port = port
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/state", self._get_state)
        app.router.add_post("/state", self._load_state)
        app.router.add_post("/state/reset", self._reset_state)
        app.router.add_get("/config", self._get_config)
        app.router.add_post("/config", self._set_config)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("Router management HTTP server listening on :%d", self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response({
            "status": "ok",
            "router_type": r.router_type,
            "workers": r.beta_learner.worker_ids,
        })

    async def _get_state(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response({
            "beta_learner": r.beta_learner.to_dict(),
            "lints_learner": r.lints_learner.to_dict(),
        })

    async def _load_state(self, request: web.Request) -> web.Response:
        r = self._router
        data = await request.json()
        if "beta_learner" in data:
            r.beta_learner.load_state(data["beta_learner"])
        if "lints_learner" in data:
            r.lints_learner.load_state(data["lints_learner"])
        logger.info("Router learner state loaded via HTTP")
        return web.json_response({"status": "loaded"})

    async def _reset_state(self, _request: web.Request) -> web.Response:
        r = self._router
        r.beta_learner.reset_all()
        r.lints_learner.reset_all()
        r.latency_tracker.reset()
        logger.info("Router learner state reset to pristine via HTTP")
        return web.json_response({"status": "reset"})

    async def _get_config(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response({
            "ts_weight": r.kv_ts_weight,
            "temperature": r.kv_load_temp,
            "cold_start_threshold": r.kv_thompson_cold_start,
            "idle_boost": r.idle_boost,
            "beta_decay": r.beta_learner.decay,
            "lints_v": r.lints_learner.v,
            "lints_forget_rate": r.lints_learner.forget_rate,
        })

    async def _set_config(self, request: web.Request) -> web.Response:
        """Hot-reload tunable router params on the live WorkloadAwareRouter instance."""
        data = await request.json()
        r = self._router
        applied = {}

        if "ts_weight" in data:
            r.kv_ts_weight = float(data["ts_weight"])
            applied["ts_weight"] = r.kv_ts_weight
        if "temperature" in data:
            r.kv_load_temp = float(data["temperature"])
            applied["temperature"] = r.kv_load_temp
        if "cold_start_threshold" in data:
            r.kv_thompson_cold_start = float(data["cold_start_threshold"])
            applied["cold_start_threshold"] = r.kv_thompson_cold_start
        if "idle_boost" in data:
            r.idle_boost = float(data["idle_boost"])
            applied["idle_boost"] = r.idle_boost
        if "beta_decay" in data:
            r.beta_learner.decay = float(data["beta_decay"])
            applied["beta_decay"] = r.beta_learner.decay
        if "lints_v" in data:
            r.lints_learner.v = float(data["lints_v"])
            applied["lints_v"] = r.lints_learner.v
        if "lints_forget_rate" in data:
            r.lints_learner.forget_rate = float(data["lints_forget_rate"])
            applied["lints_forget_rate"] = r.lints_learner.forget_rate

        logger.info("Router config hot-reloaded via HTTP: %s", applied)
        return web.json_response({"status": "applied", "params": applied})


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Parse CLI and load config
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    logger.info("Initializing Optimized Thompson Sampling Router (Prometheus metrics)")

    # Resolve block_size: env var KV_BLOCK_SIZE (set by startup script from
    # DYNAMO_KV_BLOCK_SIZE) takes precedence over config.yaml so there is a
    # single source of truth shared with workers and the frontend.
    config_block_size = get_nested(config, "infrastructure.block_size", 64)
    env_block_size_str = os.environ.get("KV_BLOCK_SIZE")
    if env_block_size_str is not None:
        env_block_size = int(env_block_size_str)
        if env_block_size != config_block_size:
            logger.warning(
                "KV_BLOCK_SIZE env var (%d) overrides config.yaml block_size (%d). "
                "Update config.yaml to match DYNAMO_KV_BLOCK_SIZE in .env to silence this warning.",
                env_block_size,
                config_block_size,
            )
        block_size = env_block_size
    else:
        block_size = config_block_size

    # Extract config values with nested access
    router = WorkloadAwareRouter(
        runtime,
        # Infrastructure
        block_size=block_size,
        router_type=str(get_nested(config, "infrastructure.router_type", "kv")).lower(),
        min_workers=get_nested(config, "infrastructure.min_workers", 1),
        # Affinity
        affinity_base=get_nested(config, "affinity.base", 0.30),
        affinity_reuse_weight=get_nested(config, "affinity.reuse_weight", 0.15),
        affinity_iat_weight=get_nested(config, "affinity.iat_weight", 0.20),
        sticky_load_floor=get_nested(config, "affinity.sticky_load_floor", 0.70),
        # Exploration
        base_ts_weight=get_nested(config, "exploration.base_ts_weight", 0.10),
        temp_base=get_nested(config, "exploration.temperature.base", 1.0),
        temp_min=get_nested(config, "exploration.temperature.min", 0.15),
        temp_max=get_nested(config, "exploration.temperature.max", 2.0),
        # Switching cost
        switch_cost_base=get_nested(config, "switching_cost.base", 0.20),
        switch_cost_reuse=get_nested(config, "switching_cost.reuse_penalty", 0.08),
        switch_cost_iat=get_nested(config, "switching_cost.iat_penalty", 0.05),
        # Load balancing
        queue_penalty_weight=get_nested(config, "load_balancing.queue_penalty_weight", 0.50),
        gpu_penalty_weight=get_nested(config, "load_balancing.gpu_penalty_weight", 1.00),
        outstanding_work_weight=get_nested(config, "load_balancing.outstanding_work_weight", 0.45),
        job_gpu_coupling_weight=get_nested(config, "load_balancing.job_gpu_coupling_weight", 0.40),
        job_queue_coupling_weight=get_nested(config, "load_balancing.job_queue_coupling_weight", 0.20),
        # Prefill
        prefill_token_scale=get_nested(config, "prefill.token_scale", 1024.0),
        prefill_weight=get_nested(config, "prefill.weight", 1.0),
        # LinTS
        lints_lambda=get_nested(config, "lints.lambda", 1.0),
        lints_v=get_nested(config, "lints.v", 0.25),
        lints_forget=get_nested(config, "lints.forget_rate", 0.995),
        # Beta bandit
        beta_decay=get_nested(config, "exploration.beta_decay", 1.0),
        # Feedback
        feedback_timeout_seconds=get_nested(config, "feedback.timeout_seconds", 120.0),
        pending_sweep_interval_seconds=get_nested(config, "feedback.sweep_interval_seconds", 5.0),
        timeout_reward=get_nested(config, "feedback.timeout_reward", 0.0),
        latency_ema_alpha=get_nested(config, "feedback.latency_ema_alpha", 0.2),
        reward_baseline_mode=get_nested(config, "feedback.reward_baseline_mode", "global"),
        # kv_load (Dynamo-native)
        kv_load_overlap_weight=get_nested(config, "kv_load.overlap_score_weight", 1.0),
        kv_load_temperature=get_nested(config, "kv_load.temperature", 0.0),
        metrics_scrape_interval=get_nested(config, "kv_thompson.metrics_scrape_interval",
                                get_nested(config, "kv_load.metrics_scrape_interval", 0.1)),
        # kv_load_balanced / kv_thompson
        idle_boost=get_nested(config, "kv_thompson.idle_boost",
                   get_nested(config, "kv_load_balanced.idle_boost", 0.02)),
        kv_load_temp=get_nested(config, "kv_thompson.temperature",
                     get_nested(config, "kv_load_balanced.temperature", 0.15)),
        kv_ts_weight=get_nested(config, "kv_thompson.ts_weight", 0.05),
        kv_thompson_queue_penalty_weight=(
            get_nested(config, "kv_thompson.queue_penalty_weight", None)
            or get_nested(config, "load_balancing.queue_penalty_weight", 1.50)
        ),
        kv_thompson_cold_start=get_nested(config, "kv_thompson.cold_start_threshold", 0.05),
        # kv_thompson feature toggles
        kt_enable_lints=get_nested(config, "kv_thompson.enable_lints", False),
        kt_enable_affinity=get_nested(config, "kv_thompson.enable_affinity", False),
        kt_enable_switching_cost=get_nested(config, "kv_thompson.enable_switching_cost", False),
        kt_enable_full_load=get_nested(config, "kv_thompson.enable_full_load", False),
        kt_enable_adaptive_temp=get_nested(config, "kv_thompson.enable_adaptive_temp", False),
        kt_enable_adaptive_explore=get_nested(config, "kv_thompson.enable_adaptive_explore", False),
        kt_enable_sticky_floor=get_nested(config, "kv_thompson.enable_sticky_floor", False),
        kt_lints_weight=get_nested(config, "kv_thompson.lints_weight", 1.0),
        kt_affinity_base=get_nested(config, "kv_thompson.affinity_base", 0.30),
        kt_affinity_reuse_weight=get_nested(config, "kv_thompson.affinity_reuse_weight", 0.08),
        kt_affinity_iat_weight=get_nested(config, "kv_thompson.affinity_iat_weight", 0.20),
        kt_switch_base=get_nested(config, "kv_thompson.switch_base", 0.10),
        kt_switch_reuse=get_nested(config, "kv_thompson.switch_reuse", 0.04),
        kt_switch_iat=get_nested(config, "kv_thompson.switch_iat", 0.03),
        kt_sticky_load_floor=get_nested(config, "kv_thompson.sticky_load_floor", 0.01),
        kt_load_mod_floor=get_nested(config, "kv_thompson.load_mod_floor", 0.0),
        kt_adaptive_temp_base=get_nested(config, "kv_thompson.adaptive_temp_base", 1.0),
        # Debug
        debug_traces=get_nested(config, "debug.traces_enabled", False),
        debug_trace_dir=get_nested(config, "debug.trace_dir", "/tmp/dynamo_router_traces"),
        debug_buffer_size=get_nested(config, "debug.buffer_size", 2000),
    )
    await router.initialize()

    # Start learner management HTTP server
    mgmt_port = int(os.environ.get("ROUTER_MGMT_PORT", "8085"))
    config_path = args.config or str(get_default_config_path())
    mgmt_server = RouterManagementServer(router, config_path=config_path, port=mgmt_port)
    await mgmt_server.start()

    # Serve both endpoints
    find_worker_ep = _get_endpoint(runtime, "dynamo", "router", "find_worker")
    feedback_ep = _get_endpoint(runtime, "dynamo", "router", "feedback")
    await asyncio.gather(
        find_worker_ep.serve_endpoint(router.generate),
        feedback_ep.serve_endpoint(router.feedback),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
