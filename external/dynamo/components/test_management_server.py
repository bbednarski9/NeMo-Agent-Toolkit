# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the LearnerManagementServer HTTP endpoint logic.

Tests the behavior of health, state (get/load/reset), and config (get/set)
by directly invoking the same logic that the endpoints use, without
requiring aiohttp, Dynamo runtime, or full ProcessorRequestHandler.

Run with:  pytest test_management_server.py -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from learners import BetaLearner, LatencyTracker, LinTSLearner


# ---------------------------------------------------------------------------
# MockHandler - matches ProcessorRequestHandler interface for mgmt endpoints
# ---------------------------------------------------------------------------
class MockHandler:
    """Stub handler with same interface as ProcessorRequestHandler for mgmt tests."""

    def __init__(self):
        self.routing_mode = "kv_thompson_native"
        self.beta_learner = BetaLearner(decay=0.995)
        self.lints_learner = LinTSLearner(
            feature_dim=6, lambda_=1.0, v=0.25, forget_rate=0.995
        )
        self.latency_tracker = LatencyTracker(ema_alpha=0.2)
        for wid in [0, 1, 2]:
            self.beta_learner.add_worker(wid)
            self.lints_learner.add_worker(wid)


# ---------------------------------------------------------------------------
# Logic extracted from LearnerManagementServer (avoids processor.py dynamo imports)
# ---------------------------------------------------------------------------
TUNABLE_PARAM_KEYS = [
    "ts_weight",
    "temperature",
    "cold_start_threshold",
    "idle_boost",
    "beta_decay",
    "lints_v",
    "lints_forget_rate",
    "queue_penalty_weight",
    "lints_weight",
]


def health_response(handler: MockHandler) -> dict:
    """Replicate _health endpoint logic."""
    return {
        "status": "ok",
        "routing_mode": handler.routing_mode,
        "has_learners": handler.beta_learner is not None,
    }


def get_state_response(handler: MockHandler) -> dict | tuple[dict, int]:
    """Replicate _get_state endpoint logic. Returns (body, status) or body for 200."""
    if handler.beta_learner is None or handler.lints_learner is None:
        return {"error": "no in-process learners (not kv_thompson_native)"}, 400
    return {
        "beta_learner": handler.beta_learner.to_dict(),
        "lints_learner": handler.lints_learner.to_dict(),
    }


def load_state(handler: MockHandler, data: dict) -> dict | tuple[dict, int]:
    """Replicate _load_state endpoint logic."""
    if handler.beta_learner is None or handler.lints_learner is None:
        return {"error": "no in-process learners"}, 400
    if "beta_learner" in data:
        handler.beta_learner.load_state(data["beta_learner"])
    if "lints_learner" in data:
        handler.lints_learner.load_state(data["lints_learner"])
    return {"status": "loaded"}


def reset_state(handler: MockHandler) -> dict | tuple[dict, int]:
    """Replicate _reset_state endpoint logic."""
    if handler.beta_learner is None or handler.lints_learner is None:
        return {"error": "no in-process learners"}, 400
    handler.beta_learner.reset_all()
    handler.lints_learner.reset_all()
    if handler.latency_tracker is not None:
        handler.latency_tracker.reset()
    return {"status": "reset"}


def get_config_response(config_path: str) -> dict | tuple[dict, int]:
    """Replicate _get_config endpoint logic."""
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        kt = cfg.get("kv_thompson", {})
        lints = cfg.get("lints", {})
        exploration = cfg.get("exploration", {})
        return {
            "ts_weight": kt.get("ts_weight"),
            "temperature": kt.get("temperature"),
            "cold_start_threshold": kt.get("cold_start_threshold"),
            "idle_boost": kt.get("idle_boost"),
            "beta_decay": exploration.get("beta_decay"),
            "lints_v": lints.get("v"),
            "lints_forget_rate": lints.get("forget_rate"),
        }
    except Exception as e:
        return {"error": str(e)}, 500


def set_config(
    handler: MockHandler, config_path: str, data: dict
) -> dict | tuple[dict, int]:
    """Replicate _set_config endpoint logic."""
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        kt = cfg.setdefault("kv_thompson", {})
        lints = cfg.setdefault("lints", {})
        exploration = cfg.setdefault("exploration", {})
        applied = {}

        if "ts_weight" in data:
            kt["ts_weight"] = float(data["ts_weight"])
            applied["ts_weight"] = kt["ts_weight"]
        if "temperature" in data:
            kt["temperature"] = float(data["temperature"])
            applied["temperature"] = kt["temperature"]
        if "cold_start_threshold" in data:
            kt["cold_start_threshold"] = float(data["cold_start_threshold"])
            applied["cold_start_threshold"] = kt["cold_start_threshold"]
        if "idle_boost" in data:
            kt["idle_boost"] = float(data["idle_boost"])
            applied["idle_boost"] = kt["idle_boost"]
        if "beta_decay" in data:
            exploration["beta_decay"] = float(data["beta_decay"])
            applied["beta_decay"] = exploration["beta_decay"]
            if handler.beta_learner is not None:
                handler.beta_learner.decay = float(data["beta_decay"])
        if "lints_v" in data:
            lints["v"] = float(data["lints_v"])
            applied["lints_v"] = lints["v"]
            if handler.lints_learner is not None:
                handler.lints_learner.v = float(data["lints_v"])
        if "lints_forget_rate" in data:
            lints["forget_rate"] = float(data["lints_forget_rate"])
            applied["lints_forget_rate"] = lints["forget_rate"]
            if handler.lints_learner is not None:
                handler.lints_learner.forget_rate = float(data["lints_forget_rate"])

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        return {"status": "applied", "params": applied}
    except Exception as e:
        return {"error": str(e)}, 500


# ===========================================================================
# TestManagementServerHealth
# ===========================================================================
class TestManagementServerHealth:
    """Tests for /health endpoint logic."""

    def test_health_returns_ok(self):
        """GET /health returns 200 with status=ok."""
        handler = MockHandler()
        resp = health_response(handler)
        assert resp["status"] == "ok"
        assert resp["routing_mode"] == "kv_thompson_native"
        assert resp["has_learners"] is True


# ===========================================================================
# TestManagementServerState
# ===========================================================================
class TestManagementServerState:
    """Tests for /state and /state/reset endpoint logic."""

    def test_get_state_returns_learner_dicts(self):
        """GET /state returns JSON with beta_learner and lints_learner keys."""
        handler = MockHandler()
        resp = get_state_response(handler)
        assert "beta_learner" in resp
        assert "lints_learner" in resp
        assert resp["beta_learner"]["type"] == "BetaLearner"
        assert resp["lints_learner"]["type"] == "LinTSLearner"
        assert "bandits" in resp["beta_learner"]
        assert "workers" in resp["lints_learner"]

    def test_post_state_loads_learners(self):
        """POST /state with JSON body loads state into learners."""
        handler = MockHandler()
        # Mutate beta learner
        for _ in range(5):
            handler.beta_learner.update(0, reward=0.9)
        # Mutate lints learner
        x = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0], dtype=np.float64)
        handler.lints_learner.update(0, x, reward=0.8)

        # Capture state, reset, then load state back
        beta_state = handler.beta_learner.to_dict()
        lints_state = handler.lints_learner.to_dict()

        reset_state(handler)
        load_state(handler, {"beta_learner": beta_state, "lints_learner": lints_state})

        # Verify state was restored
        assert handler.beta_learner.to_dict() == beta_state
        assert handler.lints_learner.to_dict() == lints_state

    def test_reset_state(self):
        """POST /state/reset resets both learners to pristine; verify get_params."""
        handler = MockHandler()
        # Mutate beta learner
        for _ in range(10):
            handler.beta_learner.update(0, reward=0.7)
        # Mutate lints learner
        x = np.ones(6, dtype=np.float64)
        handler.lints_learner.update(0, x, reward=0.6)

        reset_state(handler)

        # Beta: pristine = (1.0, 1.0)
        for wid in [0, 1, 2]:
            alpha, beta = handler.beta_learner.get_params(wid)
            assert alpha == pytest.approx(1.0)
            assert beta == pytest.approx(1.0)

        # LinTS: pristine = A = lambda*I, b = 0
        lambda_ = handler.lints_learner.lambda_
        for wid in [0, 1, 2]:
            A, b = handler.lints_learner.get_params(wid)
            expected_A = lambda_ * np.eye(6, dtype=np.float64)
            assert np.allclose(A, expected_A)
            assert np.allclose(b, np.zeros(6))

    def test_reset_also_resets_latency_tracker(self):
        """POST /state/reset clears LatencyTracker baselines."""
        handler = MockHandler()
        # Populate latency baselines
        handler.latency_tracker.update_baselines(
            wid=0, osl="short", prefill_bin="small", metric=50.0, per_tok=True
        )
        handler.latency_tracker.update_baselines(
            wid=1, osl="short", prefill_bin="small", metric=60.0, per_tok=True
        )

        # Verify baselines exist before reset
        b0 = handler.latency_tracker.get_baseline(
            wid=0, osl="short", prefill_bin="small", per_tok=True, fallback=1.0
        )
        assert b0 is not None and b0 > 0

        reset_state(handler)

        # After reset, hierarchical lookup should fall through to global → fallback
        b_after = handler.latency_tracker.get_baseline(
            wid=0, osl="short", prefill_bin="small", per_tok=True, fallback=100.0
        )
        assert b_after == 100.0  # Fallback used when no baselines
        assert len(handler.latency_tracker._worker) == 0
        assert len(handler.latency_tracker._bucket) == 0
        assert handler.latency_tracker._global[True] is None


# ===========================================================================
# TestManagementServerConfig
# ===========================================================================
class TestManagementServerConfig:
    """Tests for /config GET and POST endpoint logic."""

    def test_get_config_returns_params(self):
        """GET /config returns JSON with the 7 tunable params."""
        config_yaml = """
kv_thompson:
  ts_weight: 0.05
  temperature: 0.30
  cold_start_threshold: 0.05
  idle_boost: 0.02
lints:
  v: 0.25
  forget_rate: 0.995
exploration:
  beta_decay: 0.995
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            resp = get_config_response(config_path)
            assert set(resp.keys()) == set(TUNABLE_PARAM_KEYS)
            assert resp["ts_weight"] == 0.05
            assert resp["temperature"] == 0.30
            assert resp["cold_start_threshold"] == 0.05
            assert resp["idle_boost"] == 0.02
            assert resp["beta_decay"] == 0.995
            assert resp["lints_v"] == 0.25
            assert resp["lints_forget_rate"] == 0.995
        finally:
            os.unlink(config_path)

    def test_post_config_applies_params(self):
        """POST /config with params applies them to learner instances."""
        config_yaml = """
kv_thompson:
  ts_weight: 0.05
  temperature: 0.30
lints:
  v: 0.25
  forget_rate: 0.995
exploration:
  beta_decay: 0.995
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            handler = MockHandler()
            assert handler.beta_learner.decay == 0.995
            assert handler.lints_learner.v == 0.25
            assert handler.lints_learner.forget_rate == 0.995

            # Apply new params: beta_decay and lints_v affect learners
            resp = set_config(
                handler,
                config_path,
                {
                    "ts_weight": 0.10,
                    "temperature": 0.50,
                    "beta_decay": 0.99,
                    "lints_v": 0.40,
                    "lints_forget_rate": 0.98,
                },
            )
            assert resp["status"] == "applied"
            assert resp["params"]["beta_decay"] == 0.99
            assert resp["params"]["lints_v"] == 0.40
            assert resp["params"]["lints_forget_rate"] == 0.98

            # Verify learner instances were updated
            assert handler.beta_learner.decay == 0.99
            assert handler.lints_learner.v == 0.40
            assert handler.lints_learner.forget_rate == 0.98

            # Verify config file was written
            resp2 = get_config_response(config_path)
            assert resp2["ts_weight"] == 0.10
            assert resp2["temperature"] == 0.50
            assert resp2["beta_decay"] == 0.99
            assert resp2["lints_v"] == 0.40
            assert resp2["lints_forget_rate"] == 0.98
        finally:
            os.unlink(config_path)
