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
KvThompsonRouter — In-process Thompson Sampling router using native KvRouter (pyo3).

Uses Dynamo's native KvRouter for KV cache state (overlap scores, load signals)
and applies Thompson Sampling (Beta bandits + LinTS contextual bandits) on top
for learning-based worker selection.

All scoring features are independently togglable via config.yaml.
The router is instantiated by processor.py and called in-process — no NATS RPC.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from aiohttp import web

from learners import BetaLearner, LatencyTracker, LinTSLearner

logger = logging.getLogger(__name__)

TUNABLE_ROUTER_PARAMS = [
    "ts_weight", "temperature", "cold_start_threshold", "idle_boost",
    "beta_decay", "lints_v", "lints_forget_rate",
    "queue_penalty_weight", "lints_weight",
]


@dataclass
class RoutingDecision:
    """Result of a pick_worker() call, passed back to update_feedback()."""

    chosen: int
    native_pick: int
    features: np.ndarray | None = None
    loads_by_wid: dict[int, dict] = field(default_factory=dict)
    worker_details: list[dict] = field(default_factory=list)
    prefix_id: str = ""
    osl: int = 250
    iat: int = 250
    reuse_budget: int = 0
    tokens_in: int = 0
    last_worker: int | None = None


class KvThompsonRouter:
    """In-process Thompson Sampling router backed by native KvRouter (pyo3).

    Modular scoring with independently togglable features:

      Base scoring (always on):
        effective_overlap = max(overlap, idle_boost)     [if enable_idle_boost]
        load_mod = exp(-qpw * decode_blocks² / 2500)    [exponential queue proxy]
        score = effective_overlap * load_mod + ts_weight * Beta(wid)

      Optional features (each independently togglable):
        enable_lints           — LinTS contextual bandit (7-dim feature-aware)
        enable_affinity        — prefix stickiness for multi-turn sessions
        enable_switching_cost  — penalty for migrating prefix to different worker
        enable_adaptive_temp   — temperature decays with session depth
        enable_adaptive_explore — Beta-TS weight decays with session depth
        enable_sticky_floor    — protect sticky worker's load_mod minimum

      Selection:
        enable_softmax=true  → softmax(scores, temperature) → probabilistic pick
        enable_softmax=false → deterministic argmax

      Cold start:
        enable_cold_start=true → round-robin when max overlap < threshold

    Feature vector (7 dims):
        [bias, inv_prefill, inv_decode, affinity, osl_norm, reuse_norm,
         iat_norm]
    """

    def __init__(self, kv_router, config: dict | None = None):
        self.kv_router = kv_router
        cfg = config or {}

        kt = cfg.get("kv_thompson", {})

        # Learner hyperparameters (defaults match original hardcoded values)
        beta_decay = float(kt.get("beta_decay", 0.995))
        lints_lambda = float(kt.get("lints_lambda", 1.0))
        lints_v = float(kt.get("lints_v", 0.25))
        lints_forget = float(kt.get("lints_forget_rate", 0.995))
        latency_ema_alpha = float(kt.get("latency_ema_alpha", 0.2))

        # Base scoring parameters
        self.ts_weight = float(kt.get("ts_weight", 0.05))
        self.idle_boost = float(kt.get("idle_boost", 0.135))
        self.temperature = float(kt.get("temperature", 1.70))
        self.cold_start_threshold = float(kt.get("cold_start_threshold", 0.37))
        self.queue_penalty_weight = float(kt.get("queue_penalty_weight", 2.5))
        self.load_mod_floor = float(kt.get("load_mod_floor", 0.3))

        # Temperature bounds
        self.temp_min = float(kt.get("temp_min", 0.15))
        self.temp_max = float(kt.get("temp_max", 2.0))

        # Feature toggles (all default to False to preserve Phase 1 behavior)
        self.enable_softmax = bool(kt.get("enable_softmax", False))
        self.enable_cold_start = bool(kt.get("enable_cold_start", False))
        self.enable_idle_boost = bool(kt.get("enable_idle_boost", False))
        self.enable_load_mod_floor = bool(kt.get("enable_load_mod_floor", False))
        self.enable_lints = bool(kt.get("enable_lints", False))
        self.enable_affinity = bool(kt.get("enable_affinity", False))
        self.enable_switching_cost = bool(kt.get("enable_switching_cost", False))
        self.enable_adaptive_temp = bool(kt.get("enable_adaptive_temp", False))
        self.enable_adaptive_explore = bool(kt.get("enable_adaptive_explore", False))
        self.enable_sticky_floor = bool(kt.get("enable_sticky_floor", False))

        # Feature-specific weights
        self.lints_weight = float(kt.get("lints_weight", -1.0))
        self.affinity_base = float(kt.get("affinity_base", 0.15))
        self.affinity_reuse_weight = float(kt.get("affinity_reuse_weight", 0.02))
        self.switch_base = float(kt.get("switch_base", 0.04))
        self.switch_reuse = float(kt.get("switch_reuse", 0.01))
        self.sticky_load_floor = float(kt.get("sticky_load_floor", 0.01))
        self.adaptive_temp_base = float(kt.get("adaptive_temp_base", 1.0))

        # Initialize learners
        self.feature_dim = 7
        self.beta_learner = BetaLearner(decay=beta_decay)
        self.lints_learner = LinTSLearner(
            feature_dim=self.feature_dim,
            lambda_=lints_lambda,
            v=lints_v,
            forget_rate=lints_forget,
        )
        self.latency_tracker = LatencyTracker(ema_alpha=latency_ema_alpha)

        # Prefix → last worker mapping for affinity
        self._prefix_workers: dict[str, int] = {}
        self._cold_start_rr: int = 0

        features_on = [
            name for name, enabled in [
                ("softmax", self.enable_softmax),
                ("cold_start", self.enable_cold_start),
                ("idle_boost", self.enable_idle_boost),
                ("load_mod_floor", self.enable_load_mod_floor),
                ("lints", self.enable_lints),
                ("affinity", self.enable_affinity),
                ("switching_cost", self.enable_switching_cost),
                ("adaptive_temp", self.enable_adaptive_temp),
                ("adaptive_explore", self.enable_adaptive_explore),
                ("sticky_floor", self.enable_sticky_floor),
            ] if enabled
        ]
        logger.info(
            "KvThompsonRouter initialized (feature_dim=%d, beta_decay=%.3f, "
            "lints_v=%.3f, ts_weight=%.3f, features=[%s])",
            self.feature_dim, beta_decay, lints_v, self.ts_weight,
            ", ".join(features_on) if features_on else "base only",
        )

    async def pick_worker(
        self,
        token_ids: list[int],
        prefix_id: str,
        reuse_budget: int,
        osl: int,
        iat: int,
        tokens_in: int,
    ) -> RoutingDecision:
        """Score workers and pick the best one."""
        loads = await self.kv_router.get_potential_loads(token_ids)
        native_pick, _, _ = await self.kv_router.best_worker(token_ids)

        worker_ids: list[int] = []
        raw_scores: list[float] = []
        all_overlaps: dict[int, float] = {}
        loads_by_wid: dict[int, dict] = {}
        features_by_wid: dict[int, np.ndarray] = {}
        last_worker = self._prefix_workers.get(prefix_id)
        worker_details: list[dict] = []

        iat_factor = self._iat_factor(iat)

        for load_info in loads:
            wid = load_info["worker_id"]
            prefill_tokens = load_info.get("potential_prefill_tokens", 0)
            decode_blocks = load_info.get("potential_decode_blocks", 0)
            worker_ids.append(wid)
            loads_by_wid[wid] = load_info

            self.beta_learner.add_worker(wid)
            self.lints_learner.add_worker(wid)

            x = self._build_features(
                prefill_tokens, decode_blocks, last_worker, wid,
                osl, reuse_budget, tokens_in, iat,
            )
            features_by_wid[wid] = x

            overlap = 1.0 - prefill_tokens / max(1, tokens_in)
            all_overlaps[wid] = overlap

            score = self._score_worker(
                wid, x, overlap, decode_blocks, last_worker, reuse_budget, iat_factor,
            )
            raw_scores.append(score)

            worker_details.append({
                "id": wid,
                "kv_overlap": round(overlap, 4),
                "prefill_tokens": prefill_tokens,
                "decode_blocks": decode_blocks,
                "beta_sample": round(self.beta_learner.sample(wid), 4),
                "lints_sample": round(math.tanh(self.lints_learner.sample(wid, x)), 4),
                "final_score": round(score, 4),
            })

        # --- Selection ---
        if not worker_ids:
            chosen = native_pick
        elif self.enable_cold_start:
            best_overlap = max(all_overlaps.values()) if all_overlaps else 0.0
            if best_overlap < self.cold_start_threshold:
                idx = self._cold_start_rr % len(worker_ids)
                self._cold_start_rr += 1
                chosen = worker_ids[idx]
                logger.info(
                    "COLD_START: prefix=%s chosen=%s best_ov=%.4f threshold=%.4f rr_idx=%d/%d",
                    prefix_id, chosen, best_overlap,
                    self.cold_start_threshold, idx, len(worker_ids),
                )
            else:
                chosen = self._select_from_scores(worker_ids, raw_scores, reuse_budget, iat_factor)
        else:
            chosen = self._select_from_scores(worker_ids, raw_scores, reuse_budget, iat_factor)

        self._prefix_workers[prefix_id] = chosen

        return RoutingDecision(
            chosen=chosen,
            native_pick=native_pick,
            features=features_by_wid.get(chosen),
            loads_by_wid=loads_by_wid,
            worker_details=worker_details,
            prefix_id=prefix_id,
            osl=osl,
            iat=iat,
            reuse_budget=reuse_budget,
            tokens_in=tokens_in,
            last_worker=last_worker,
        )

    def _select_from_scores(
        self,
        worker_ids: list[int],
        raw_scores: list[float],
        reuse_budget: int,
        iat_factor: float,
    ) -> int:
        if self.enable_softmax:
            if self.enable_adaptive_temp:
                temp = self.adaptive_temp_base / (1.0 + float(reuse_budget) * iat_factor)
                temp = min(max(temp, self.temp_min), self.temp_max)
            else:
                temp = self.temperature
            probs = self._softmax(raw_scores, temp)
            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    return worker_ids[i]
            return worker_ids[-1]
        else:
            best_idx = int(np.argmax(raw_scores))
            return worker_ids[best_idx]

    def _softmax(self, scores: list[float], temp: float) -> list[float]:
        t = float(min(max(temp, self.temp_min), self.temp_max))
        arr = np.array(scores)
        m = float(np.max(arr))
        exps = np.exp((arr - m) / max(1e-6, t))
        s = float(np.sum(exps))
        if s <= 0.0 or not np.isfinite(s):
            return [1.0 / len(scores)] * len(scores)
        return list((exps / s).astype(float))

    def update_feedback(
        self,
        decision: RoutingDecision,
        latency_ms: float,
        tokens_out: int,
    ) -> dict[str, Any]:
        """Update learners with observed latency reward."""
        metric, per_tok = LatencyTracker.latency_metric(latency_ms, tokens_out)
        baseline = self.latency_tracker.get_global_baseline(per_tok, fallback=metric)
        reward = LatencyTracker.compute_reward(metric, baseline, True)

        self.beta_learner.update(decision.chosen, reward)

        x_chosen = decision.features
        if x_chosen is None:
            chosen_load = decision.loads_by_wid.get(decision.chosen, {})
            x_chosen = self._build_features(
                chosen_load.get("potential_prefill_tokens", 0),
                chosen_load.get("potential_decode_blocks", 0),
                decision.last_worker,
                decision.chosen,
                decision.osl,
                decision.reuse_budget,
                decision.tokens_in,
                decision.iat,
            )
        self.lints_learner.update(decision.chosen, x_chosen, reward)
        self.latency_tracker.update_baselines(decision.chosen, "M", "L", metric, per_tok)

        beta_alpha, beta_beta = self.beta_learner.get_params(decision.chosen)
        lints_mean = self.lints_learner.posterior_mean(decision.chosen).tolist()

        logger.debug(
            "Feedback: wid=%s metric=%.2f baseline=%.2f reward=%.3f tokens_out=%d",
            decision.chosen, metric, baseline, reward, tokens_out,
        )

        return {
            "metric": metric,
            "baseline_ema": baseline,
            "reward": reward,
            "beta_after": {"alpha": round(beta_alpha, 4), "beta": round(beta_beta, 4)},
            "lints_posterior_mean": [round(v, 6) for v in lints_mean],
        }

    # -------------------- Scoring -------------------- #

    def _score_worker(
        self,
        wid: int,
        x: np.ndarray,
        overlap: float,
        decode_blocks: int,
        last_worker: int | None,
        reuse_budget: int,
        iat_factor: float,
    ) -> float:
        inv_prefill = float(x[1])
        inv_decode = float(x[2])

        # Base score: original formula (inv_prefill + inv_decode weighted sum)
        base_score = inv_prefill * 0.5 + inv_decode * 0.3
        score = base_score

        # Beta-TS exploration
        if self.enable_adaptive_explore:
            ts_w_eff = self.ts_weight / (1.0 + float(reuse_budget) * iat_factor)
        else:
            ts_w_eff = self.ts_weight
        score += ts_w_eff * self.beta_learner.sample(wid)

        # LinTS contextual bandit
        if self.enable_lints:
            raw_lints = self.lints_learner.sample(wid, x)
            if self.lints_weight < 0:
                score += abs(self.lints_weight) * math.tanh(raw_lints)
            else:
                score += self.lints_weight * raw_lints
        else:
            score += math.tanh(self.lints_learner.sample(wid, x))

        # --- Phase 2 features (only active when toggled on) ---

        if self.enable_idle_boost or self.enable_load_mod_floor or self.enable_sticky_floor:
            effective_overlap = max(overlap, self.idle_boost) if self.enable_idle_boost else overlap
            qpw = self.queue_penalty_weight
            db = max(0.0, float(decode_blocks))
            load_mod = math.exp(-qpw * db * db / 2500.0)

            if self.enable_sticky_floor and last_worker == wid and reuse_budget > 0:
                load_mod = max(load_mod, self.sticky_load_floor)
            if self.enable_load_mod_floor and self.load_mod_floor > 0.0:
                load_mod = max(load_mod, self.load_mod_floor)

            score += effective_overlap * load_mod

        if self.enable_affinity and last_worker == wid and reuse_budget > 0:
            score += (self.affinity_base
                      + self.affinity_reuse_weight * float(reuse_budget)) * (0.5 + 0.5 * overlap)

        if self.enable_switching_cost and last_worker is not None and wid != last_worker and reuse_budget > 0:
            score -= (self.switch_base + self.switch_reuse * float(reuse_budget))

        if np.isnan(score) or np.isinf(score):
            score = -1e9

        return float(score)

    # -------------------- Feature Vector -------------------- #

    @staticmethod
    def _decode_cost(osl: int) -> float:
        """Interpolate decode cost from continuous OSL (tokens).

        Anchor points: 128->1.0, 250->2.0, 1024->3.0.
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
        """Interpolate IAT factor from continuous IAT (ms).

        Anchor points: 50->1.5, 250->1.0, 1000->0.6.
        """
        if iat <= 50:
            return 1.5
        if iat <= 250:
            return 1.5 - 0.5 * (iat - 50) / (250 - 50)
        if iat >= 1000:
            return 0.6
        return 1.0 - 0.4 * (iat - 250) / (1000 - 250)

    def _build_features(
        self,
        prefill_tokens: int,
        decode_blocks: int,
        last_worker: int | None,
        wid: int,
        osl: int,
        reuse_budget: int,
        tokens_in: int,
        iat: int,
    ) -> np.ndarray:
        inv_prefill = 1.0 / (1.0 + prefill_tokens / 1000.0)
        inv_decode = 1.0 / (1.0 + decode_blocks / 50.0)
        affinity = 1.0 if (last_worker is not None and wid == last_worker) else 0.0
        osl_norm = min(osl, 1024) / 1024.0
        reuse_norm = math.tanh(0.25 * max(reuse_budget, 0))
        iat_norm = (self._iat_factor(iat) - 0.6) / 0.9
        return np.array(
            [1.0, inv_prefill, inv_decode, affinity, osl_norm, reuse_norm,
             iat_norm],
            dtype=np.float64,
        )

    # -------------------- Management HTTP Server -------------------- #


class RouterManagementServer:
    """HTTP server for learner state persistence, config hot-reload, and reset.

    All tunable params are applied directly to the live KvThompsonRouter instance.
    """

    def __init__(self, router: KvThompsonRouter, port: int = 8084):
        self._router = router
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
        logger.info("RouterManagementServer listening on :%d", self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response({
            "status": "ok",
            "router_type": "kv_thompson",
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
        logger.info("Learner state loaded via HTTP")
        return web.json_response({"status": "loaded"})

    async def _reset_state(self, _request: web.Request) -> web.Response:
        r = self._router
        r.beta_learner.reset_all()
        r.lints_learner.reset_all()
        r.latency_tracker.reset()
        logger.info("Learner state reset to pristine via HTTP")
        return web.json_response({"status": "reset"})

    async def _get_config(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response({
            "ts_weight": r.ts_weight,
            "temperature": r.temperature,
            "cold_start_threshold": r.cold_start_threshold,
            "idle_boost": r.idle_boost,
            "beta_decay": r.beta_learner.decay,
            "lints_v": r.lints_learner.v,
            "lints_forget_rate": r.lints_learner.forget_rate,
            "queue_penalty_weight": r.queue_penalty_weight,
            "lints_weight": r.lints_weight,
        })

    async def _set_config(self, request: web.Request) -> web.Response:
        """Hot-reload tunable params directly onto the live router instance."""
        data = await request.json()
        r = self._router
        applied = {}

        if "ts_weight" in data:
            r.ts_weight = float(data["ts_weight"])
            applied["ts_weight"] = r.ts_weight
        if "temperature" in data:
            r.temperature = float(data["temperature"])
            applied["temperature"] = r.temperature
        if "cold_start_threshold" in data:
            r.cold_start_threshold = float(data["cold_start_threshold"])
            applied["cold_start_threshold"] = r.cold_start_threshold
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
        if "queue_penalty_weight" in data:
            r.queue_penalty_weight = float(data["queue_penalty_weight"])
            applied["queue_penalty_weight"] = r.queue_penalty_weight
        if "lints_weight" in data:
            r.lints_weight = float(data["lints_weight"])
            applied["lints_weight"] = r.lints_weight

        logger.info("Router config hot-reloaded via HTTP: %s", applied)
        return web.json_response({"status": "applied", "params": applied})
