"""
End-to-end tests for Thompson Sampling router learner components.

Tests cover:
  - BetaLearner:      Bayesian bandit convergence, parameter ranges, multi-worker isolation
  - LinTSLearner:     Contextual bandit learning, forgetting, numerical stability
  - LatencyTracker:   EMA baselines, hierarchical lookup, reward computation
  - PendingDecisions: Lifecycle, timeout sweep, thread safety
  - Integration:      Full feedback flow (decision → pending → feedback → learner update)

Run with:  pytest test_thompson_learners.py -v
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from learners import BetaLearner, LatencyTracker, LinTSLearner, PendingDecisions


# ===========================================================================
# BetaLearner
# ===========================================================================
class TestBetaLearner:
    """Tests for the per-worker Beta-Thompson Sampling bandit."""

    def test_initial_uniform_prior(self):
        bl = BetaLearner()
        bl.add_worker(1)
        alpha, beta = bl.get_params(1)
        assert alpha == 1.0
        assert beta == 1.0
        assert bl.mean(1) == pytest.approx(0.5)

    def test_positive_reward_increases_alpha(self):
        bl = BetaLearner()
        bl.add_worker(1)
        bl.update(1, reward=0.8)
        alpha, beta = bl.get_params(1)
        assert alpha == pytest.approx(1.8)
        assert beta == pytest.approx(1.2)

    def test_zero_reward_increases_beta(self):
        bl = BetaLearner()
        bl.add_worker(1)
        bl.update(1, reward=0.0)
        alpha, beta = bl.get_params(1)
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(2.0)

    def test_reward_clamped_to_unit_interval(self):
        bl = BetaLearner()
        bl.add_worker(1)
        # Reward > 1 should be clamped to 1.0
        bl.update(1, reward=5.0)
        alpha, beta = bl.get_params(1)
        assert alpha == pytest.approx(2.0)
        assert beta == pytest.approx(1.0)
        # Reward < 0 should be clamped to 0.0
        bl.update(1, reward=-3.0)
        alpha, beta = bl.get_params(1)
        assert alpha == pytest.approx(2.0)
        assert beta == pytest.approx(2.0)

    def test_convergence_high_reward(self):
        """After many reward=1 updates, posterior mean should approach 1."""
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(200):
            bl.update(1, reward=1.0)
        assert bl.mean(1) > 0.95

    def test_convergence_low_reward(self):
        """After many reward=0 updates, posterior mean should approach 0."""
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(200):
            bl.update(1, reward=0.0)
        assert bl.mean(1) < 0.05

    def test_convergence_mixed_reward(self):
        """With 70% reward=1 and 30% reward=0, mean should converge near 0.7."""
        bl = BetaLearner()
        bl.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(500):
            r = 1.0 if rng.random() < 0.7 else 0.0
            bl.update(1, r)
        assert bl.mean(1) == pytest.approx(0.7, abs=0.05)

    def test_sample_in_unit_interval(self):
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(100):
            s = bl.sample(1)
            assert 0.0 <= s <= 1.0

    def test_sample_distribution_shifts_with_reward(self):
        """High-reward worker should produce higher samples than low-reward."""
        bl = BetaLearner()
        bl.add_worker(1)
        bl.add_worker(2)
        for _ in range(100):
            bl.update(1, reward=0.9)
            bl.update(2, reward=0.1)
        high_samples = [bl.sample(1) for _ in range(500)]
        low_samples = [bl.sample(2) for _ in range(500)]
        assert np.mean(high_samples) > np.mean(low_samples) + 0.3

    def test_multi_worker_independence(self):
        """Updates to one worker should not affect another."""
        bl = BetaLearner()
        bl.add_worker(1)
        bl.add_worker(2)
        bl.update(1, reward=0.9)
        alpha1, beta1 = bl.get_params(1)
        alpha2, beta2 = bl.get_params(2)
        assert alpha1 != alpha2
        assert beta2 == 1.0

    def test_remove_worker(self):
        bl = BetaLearner()
        bl.add_worker(1)
        bl.update(1, reward=0.5)
        bl.remove_worker(1)
        # After removal, get_params returns default (1, 1)
        alpha, beta = bl.get_params(1)
        assert alpha == 1.0 and beta == 1.0

    def test_reset_worker(self):
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(50):
            bl.update(1, reward=1.0)
        bl.reset(1, alpha=1.0, beta=1.0)
        assert bl.mean(1) == pytest.approx(0.5)

    def test_unregistered_worker_returns_uniform(self):
        bl = BetaLearner()
        s = bl.sample(999)
        assert 0.0 <= s <= 1.0
        assert bl.mean(999) == pytest.approx(0.5)

    def test_thread_safety(self):
        """Concurrent updates should not corrupt state."""
        bl = BetaLearner()
        bl.add_worker(1)
        errors = []

        def update_many():
            try:
                for _ in range(1000):
                    bl.update(1, reward=0.5)
                    bl.sample(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_many) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        alpha, beta = bl.get_params(1)
        expected_total = 1.0 + 8000 * 0.5
        assert alpha == pytest.approx(expected_total, rel=0.01)


# ===========================================================================
# LinTSLearner
# ===========================================================================
class TestLinTSLearner:
    """Tests for the Linear Thompson Sampling contextual bandit."""

    @pytest.fixture
    def lts(self):
        return LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)

    def test_initial_state(self, lts):
        lts.add_worker(1)
        A, b = lts.get_params(1)
        assert A.shape == (4, 4)
        assert b.shape == (4,)
        np.testing.assert_allclose(A, np.eye(4))
        np.testing.assert_allclose(b, np.zeros(4))

    def test_update_modifies_A_and_b(self, lts):
        lts.add_worker(1)
        x = np.array([1.0, 0.5, 0.0, 0.0])
        lts.update(1, x, reward=0.8)
        A, b = lts.get_params(1)
        assert not np.allclose(A, np.eye(4))
        assert not np.allclose(b, np.zeros(4))

    def test_A_stays_symmetric(self, lts):
        lts.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.normal(size=4)
            lts.update(1, x, reward=rng.random())
        A, _ = lts.get_params(1)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_A_stays_positive_definite(self, lts):
        lts.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(100):
            x = rng.normal(size=4)
            lts.update(1, x, reward=rng.random())
        A, _ = lts.get_params(1)
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0)

    def test_forget_rate_decays_old_observations(self, lts):
        lts.add_worker(1)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        lts.update(1, x, reward=1.0)
        _, b_after_one = lts.get_params(1)
        # Apply many updates with zero reward in a different direction —
        # the original observation should decay away
        x2 = np.array([0.0, 1.0, 0.0, 0.0])
        for _ in range(500):
            lts.update(1, x2, reward=0.0)
        _, b_after_decay = lts.get_params(1)
        # First component (from original x) should have decayed significantly
        assert abs(b_after_decay[0]) < abs(b_after_one[0]) * 0.1

    def test_posterior_mean_learns_direction(self):
        """LinTS should learn that feature[1] predicts high reward."""
        lts = LinTSLearner(feature_dim=3, lambda_=0.1, v=0.1, forget_rate=0.999)
        lts.add_worker(1)
        rng = np.random.default_rng(123)
        for _ in range(200):
            x = rng.normal(size=3)
            x[0] = 1.0  # intercept
            reward = 1.0 / (1.0 + np.exp(-2.0 * x[1]))  # reward depends on x[1]
            lts.update(1, x, min(1.0, max(0.0, reward)))
        theta = lts.posterior_mean(1)
        # theta[1] should be positive (feature[1] correlates with reward)
        assert theta[1] > 0.1

    def test_sample_returns_float(self, lts):
        lts.add_worker(1)
        x = np.array([1.0, 0.5, 0.0, 0.0])
        s = lts.sample(1, x)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_sample_variance_decreases_with_data(self, lts):
        """More data should reduce posterior uncertainty (tighter samples)."""
        lts.add_worker(1)
        x = np.array([1.0, 0.5, 0.3, 0.1])
        early_samples = [lts.sample(1, x) for _ in range(100)]
        # Feed consistent data
        for _ in range(100):
            lts.update(1, x, reward=0.5)
        late_samples = [lts.sample(1, x) for _ in range(100)]
        assert np.std(late_samples) < np.std(early_samples)

    def test_reward_clamped_to_unit(self, lts):
        """Rewards outside [0, 1] should be clamped."""
        lts.add_worker(1)
        x = np.ones(4)
        lts.update(1, x, reward=5.0)
        _, b = lts.get_params(1)
        # With reward clamped to 1.0: b should be x * 1.0
        np.testing.assert_allclose(b, x, atol=0.1)

    def test_auto_creates_worker(self, lts):
        """Sampling/updating an unknown worker should auto-create it."""
        x = np.ones(4)
        s = lts.sample(999, x)
        assert math.isfinite(s)

    def test_multi_worker_isolation(self, lts):
        lts.add_worker(1)
        lts.add_worker(2)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        lts.update(1, x, reward=1.0)
        _, b1 = lts.get_params(1)
        _, b2 = lts.get_params(2)
        assert not np.allclose(b1, b2)

    def test_numerical_stability_zero_features(self, lts):
        lts.add_worker(1)
        x = np.zeros(4)
        s = lts.sample(1, x)
        assert math.isfinite(s)
        lts.update(1, x, reward=0.5)

    def test_thread_safety(self, lts):
        lts.add_worker(1)
        errors = []

        def work():
            try:
                rng = np.random.default_rng()
                for _ in range(200):
                    x = rng.normal(size=4)
                    lts.update(1, x, reward=rng.random())
                    lts.sample(1, x)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ===========================================================================
# LatencyTracker
# ===========================================================================
class TestLatencyTracker:
    """Tests for hierarchical EMA latency baselines and reward computation."""

    def test_ema_first_observation(self):
        lt = LatencyTracker(ema_alpha=0.2)
        val = lt.update_baselines(wid=1, osl="MEDIUM", prefill_bin="LOW", metric=100.0, per_tok=True)
        assert val == pytest.approx(100.0)

    def test_ema_smoothing(self):
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "M", "L", 100.0, True)
        val = lt.update_baselines(1, "M", "L", 200.0, True)
        # 0.2 * 200 + 0.8 * 100 = 120
        assert val == pytest.approx(120.0)

    def test_hierarchical_fallback_bucket_to_worker(self):
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "LOW", "LOW", 50.0, True)
        # Query a different bucket that doesn't exist — should fall through to worker
        val = lt.get_baseline(1, "HIGH", "HIGH", True, fallback=999.0)
        assert val == pytest.approx(50.0)

    def test_hierarchical_fallback_worker_to_global(self):
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "M", "M", 75.0, False)
        # Query a different worker — should fall through to global
        val = lt.get_baseline(2, "M", "M", False, fallback=999.0)
        assert val == pytest.approx(75.0)

    def test_hierarchical_fallback_global_to_fallback(self):
        lt = LatencyTracker(ema_alpha=0.2)
        val = lt.get_baseline(1, "M", "M", True, fallback=500.0)
        assert val == pytest.approx(500.0)

    def test_fallback_clamps_to_1(self):
        lt = LatencyTracker(ema_alpha=0.2)
        val = lt.get_baseline(1, "M", "M", True, fallback=0.001)
        assert val >= 1.0

    # --- get_global_baseline ---
    def test_global_baseline_returns_global_ema(self):
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "M", "L", 100.0, True)
        lt.update_baselines(2, "M", "L", 500.0, True)
        # Global EMA should be a mix; get_global_baseline ignores per-worker
        val = lt.get_global_baseline(True, fallback=999.0)
        assert val != pytest.approx(100.0)  # Not worker 1's baseline
        assert val != pytest.approx(500.0)  # Not worker 2's baseline
        # 0.2*500 + 0.8*100 = 180
        assert val == pytest.approx(180.0)

    def test_global_baseline_ignores_per_worker(self):
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "M", "L", 100.0, True)
        # get_baseline for worker 1 returns per-worker (100), but
        # get_global_baseline returns global (also 100 after first obs)
        assert lt.get_global_baseline(True, fallback=999.0) == pytest.approx(100.0)
        # Now update a different worker with very different latency
        lt.update_baselines(2, "M", "L", 1000.0, True)
        # get_baseline for worker 1 still returns 100 (per-worker)
        assert lt.get_baseline(1, "M", "L", True, fallback=999.0) == pytest.approx(100.0)
        # get_global_baseline returns the global mix
        global_val = lt.get_global_baseline(True, fallback=999.0)
        assert global_val > 100.0  # Shifted by worker 2's high latency

    def test_global_baseline_fallback_when_no_data(self):
        lt = LatencyTracker(ema_alpha=0.2)
        val = lt.get_global_baseline(True, fallback=500.0)
        assert val == pytest.approx(500.0)

    def test_global_baseline_fallback_clamps_to_1(self):
        lt = LatencyTracker(ema_alpha=0.2)
        val = lt.get_global_baseline(True, fallback=0.001)
        assert val >= 1.0

    # --- latency_metric ---
    def test_latency_metric_with_tokens(self):
        metric, per_tok = LatencyTracker.latency_metric(500.0, 100)
        assert per_tok is True
        assert metric == pytest.approx(5.0)

    def test_latency_metric_without_tokens(self):
        metric, per_tok = LatencyTracker.latency_metric(500.0, None)
        assert per_tok is False
        assert metric == pytest.approx(500.0)

    def test_latency_metric_zero_tokens(self):
        metric, per_tok = LatencyTracker.latency_metric(500.0, 0)
        assert per_tok is False
        assert metric == pytest.approx(500.0)

    # --- compute_reward ---
    def test_reward_success_at_baseline(self):
        """When metric == baseline, reward = 0.5."""
        r = LatencyTracker.compute_reward(metric=100.0, baseline=100.0, success=True)
        assert r == pytest.approx(0.5)

    def test_reward_fast_request(self):
        """metric << baseline → reward approaches 1."""
        r = LatencyTracker.compute_reward(metric=1.0, baseline=100.0, success=True)
        assert r > 0.95

    def test_reward_slow_request(self):
        """metric >> baseline → reward approaches 0."""
        r = LatencyTracker.compute_reward(metric=10000.0, baseline=100.0, success=True)
        assert r < 0.05

    def test_reward_failure(self):
        r = LatencyTracker.compute_reward(metric=1.0, baseline=100.0, success=False)
        assert r == 0.0

    def test_reward_always_in_unit_interval(self):
        for metric in [0.001, 1.0, 100.0, 1e6]:
            for baseline in [0.001, 1.0, 100.0]:
                r = LatencyTracker.compute_reward(metric, baseline, True)
                assert 0.0 <= r <= 1.0, f"reward={r} for metric={metric}, baseline={baseline}"

    def test_reward_monotonically_decreasing_with_metric(self):
        rewards = [LatencyTracker.compute_reward(m, 100.0, True) for m in [10, 50, 100, 500, 1000]]
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1]


# ===========================================================================
# PendingDecisions
# ===========================================================================
class TestPendingDecisions:
    """Tests for in-flight decision tracking with timeout sweep."""

    def test_add_and_pop(self):
        pd = PendingDecisions(timeout_seconds=60.0)
        pd.add("d1", {"wid": 1, "start_ts": time.time()})
        assert pd.count() == 1
        rec = pd.pop("d1")
        assert rec is not None
        assert rec["wid"] == 1
        assert pd.count() == 0

    def test_pop_unknown_returns_none(self):
        pd = PendingDecisions()
        assert pd.pop("nonexistent") is None

    def test_pop_removes_entry(self):
        pd = PendingDecisions()
        pd.add("d1", {"wid": 1, "start_ts": time.time()})
        pd.pop("d1")
        assert pd.pop("d1") is None

    def test_sweep_expires_old_decisions(self):
        pd = PendingDecisions(timeout_seconds=1.0, sweep_interval_seconds=0.0)
        now = time.time()
        pd.add("old", {"wid": 1, "start_ts": now - 2.0, "x": np.zeros(4)})
        pd.add("new", {"wid": 2, "start_ts": now, "x": np.zeros(4)})
        expired = pd.sweep(now)
        assert len(expired) == 1
        assert expired[0][0] == "old"
        assert pd.count() == 1  # "new" still pending

    def test_sweep_respects_interval(self):
        pd = PendingDecisions(timeout_seconds=1.0, sweep_interval_seconds=10.0)
        now = time.time()
        pd.add("old", {"wid": 1, "start_ts": now - 2.0})
        # First sweep should work
        expired1 = pd.sweep(now)
        # Second sweep immediately after should be skipped (interval not elapsed)
        pd.add("old2", {"wid": 2, "start_ts": now - 2.0})
        expired2 = pd.sweep(now + 0.1)
        assert len(expired1) == 1
        assert len(expired2) == 0

    def test_sweep_after_interval_elapses(self):
        pd = PendingDecisions(timeout_seconds=1.0, sweep_interval_seconds=5.0)
        now = time.time()
        pd.add("old", {"wid": 1, "start_ts": now - 2.0})
        pd.sweep(now)  # consumes "old"
        pd.add("old2", {"wid": 2, "start_ts": now - 2.0})
        expired = pd.sweep(now + 6.0)  # interval has elapsed
        assert len(expired) == 1
        assert expired[0][0] == "old2"

    def test_per_worker_counts(self):
        pd = PendingDecisions()
        pd.add("d1", {"wid": 1, "start_ts": time.time()})
        pd.add("d2", {"wid": 1, "start_ts": time.time()})
        pd.add("d3", {"wid": 2, "start_ts": time.time()})
        counts = pd.per_worker_counts()
        assert counts[1] == 2
        assert counts[2] == 1

    def test_thread_safety(self):
        pd = PendingDecisions(timeout_seconds=0.01, sweep_interval_seconds=0.0)
        errors = []

        def writer():
            try:
                for i in range(500):
                    pd.add(f"d-{threading.current_thread().name}-{i}",
                           {"wid": i % 4, "start_ts": time.time(), "x": np.zeros(4)})
            except Exception as e:
                errors.append(e)

        def sweeper():
            try:
                for _ in range(500):
                    pd.sweep(time.time())
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, name=f"w{i}") for i in range(4)]
        threads.append(threading.Thread(target=sweeper))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ===========================================================================
# Integration: End-to-End Feedback Flow
# ===========================================================================
class TestFeedbackFlowIntegration:
    """Tests simulating the full decision → pending → feedback → update cycle."""

    def test_full_feedback_updates_learners(self):
        """Simulate: router makes decision, stores pending, receives feedback, updates learners."""
        beta = BetaLearner()
        lints = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        tracker = LatencyTracker(ema_alpha=0.2)
        pending = PendingDecisions(timeout_seconds=120.0, sweep_interval_seconds=5.0)

        wid = 1
        beta.add_worker(wid)
        lints.add_worker(wid)

        x = np.array([1.0, 0.5, 0.3, 0.1])

        # 1. Router makes decision and stores in pending
        decision_id = "test-decision-001"
        now = time.time()
        pending.add(decision_id, {
            "wid": wid,
            "x": x,
            "start_ts": now,
            "osl": "MEDIUM",
            "prefill_bin": "LOW",
        })

        # 2. Feedback arrives
        latency_ms = 500.0
        tokens_out = 100
        metric, per_tok = LatencyTracker.latency_metric(latency_ms, tokens_out)
        baseline = tracker.get_baseline(wid, "MEDIUM", "LOW", per_tok, fallback=metric)
        reward = LatencyTracker.compute_reward(metric, baseline, success=True)

        # 3. Pop pending decision
        decision = pending.pop(decision_id)
        assert decision is not None
        assert decision["wid"] == wid

        # 4. Update learners
        old_alpha, old_beta = beta.get_params(wid)
        beta.update(wid, reward)
        lints.update(wid, x, reward)
        tracker.update_baselines(wid, "MEDIUM", "LOW", metric, per_tok)

        # 5. Verify beta learner updated
        new_alpha, new_beta = beta.get_params(wid)
        assert new_alpha == pytest.approx(old_alpha + reward)
        assert new_beta == pytest.approx(old_beta + 1.0 - reward)

        # 6. Verify LinTS b vector shifted toward x * reward
        _, b = lints.get_params(wid)
        np.testing.assert_allclose(b, x * reward, atol=0.1)

        # 7. Verify baseline updated
        new_baseline = tracker.get_baseline(wid, "MEDIUM", "LOW", per_tok, fallback=999.0)
        assert new_baseline == pytest.approx(metric)

    def test_timeout_flow_penalizes_learners(self):
        """Decision times out → sweep applies penalty reward to both learners."""
        beta = BetaLearner()
        lints = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        pending = PendingDecisions(timeout_seconds=1.0, sweep_interval_seconds=0.0)

        wid = 1
        beta.add_worker(wid)
        lints.add_worker(wid)
        x = np.array([1.0, 0.5, 0.3, 0.1])

        now = time.time()
        pending.add("timeout-test", {
            "wid": wid,
            "x": x,
            "start_ts": now - 2.0,  # already expired
        })

        old_alpha, old_beta_val = beta.get_params(wid)
        expired = pending.sweep(now)
        assert len(expired) == 1

        timeout_reward = 0.0
        for _, rec in expired:
            beta.update(rec["wid"], timeout_reward)
            lints.update(rec["wid"], rec["x"], timeout_reward)

        new_alpha, new_beta_val = beta.get_params(wid)
        # reward=0 → alpha unchanged, beta += 1
        assert new_alpha == pytest.approx(old_alpha)
        assert new_beta_val == pytest.approx(old_beta_val + 1.0)

    def test_repeated_good_feedback_increases_worker_preference(self):
        """Worker with consistently fast responses should get higher beta samples."""
        beta = BetaLearner()
        tracker = LatencyTracker(ema_alpha=0.2)
        beta.add_worker(1)
        beta.add_worker(2)

        # Worker 1: fast (50ms for 100 tokens = 0.5ms/tok)
        # Worker 2: slow (500ms for 100 tokens = 5.0ms/tok)
        for _ in range(100):
            m1, pt = LatencyTracker.latency_metric(50.0, 100)
            b1 = tracker.get_baseline(1, "M", "L", pt, fallback=m1)
            r1 = LatencyTracker.compute_reward(m1, b1, True)
            beta.update(1, r1)
            tracker.update_baselines(1, "M", "L", m1, pt)

            m2, pt = LatencyTracker.latency_metric(500.0, 100)
            b2 = tracker.get_baseline(2, "M", "L", pt, fallback=m2)
            r2 = LatencyTracker.compute_reward(m2, b2, True)
            beta.update(2, r2)
            tracker.update_baselines(2, "M", "L", m2, pt)

        # Worker 1 should have higher mean than worker 2
        assert beta.mean(1) > beta.mean(2)

    def test_reward_range_under_various_latencies(self):
        """All computed rewards should be in [0, 1] regardless of latency."""
        tracker = LatencyTracker(ema_alpha=0.2)
        tracker.update_baselines(1, "M", "L", 100.0, True)
        for lat_ms in [0.1, 1, 10, 100, 1000, 10000, 100000]:
            metric, per_tok = LatencyTracker.latency_metric(lat_ms, 50)
            baseline = tracker.get_baseline(1, "M", "L", per_tok, fallback=100.0)
            reward = LatencyTracker.compute_reward(metric, baseline, True)
            assert 0.0 <= reward <= 1.0, f"reward={reward} at latency={lat_ms}"

    def test_lints_learns_fast_worker_feature(self):
        """LinTS should learn to score a feature correlated with speed higher."""
        lints = LinTSLearner(feature_dim=3, lambda_=0.1, v=0.05, forget_rate=0.999)
        tracker = LatencyTracker(ema_alpha=0.2)
        lints.add_worker(1)

        rng = np.random.default_rng(42)
        for _ in range(300):
            x = np.array([1.0, rng.random(), rng.random()])
            # Feature x[1] predicts fast latency → high reward
            latency = 200.0 - 150.0 * x[1] + rng.normal(0, 10)
            metric, pt = LatencyTracker.latency_metric(max(1, latency), 100)
            baseline = tracker.get_baseline(1, "M", "L", pt, fallback=metric)
            reward = LatencyTracker.compute_reward(metric, baseline, True)
            reward = max(0.0, min(1.0, reward))
            lints.update(1, x, reward)
            tracker.update_baselines(1, "M", "L", metric, pt)

        theta = lints.posterior_mean(1)
        # theta[1] should be positive — feature[1] is correlated with reward
        assert theta[1] > 0, f"theta={theta}, expected theta[1] > 0"

    def test_concurrent_decision_feedback_cycle(self):
        """Multiple threads making decisions and sending feedback concurrently."""
        beta = BetaLearner()
        lints = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        tracker = LatencyTracker(ema_alpha=0.2)
        pending = PendingDecisions(timeout_seconds=120.0, sweep_interval_seconds=5.0)

        for wid in range(4):
            beta.add_worker(wid)
            lints.add_worker(wid)

        errors = []
        completed = {"count": 0}
        lock = threading.Lock()

        def decision_feedback_cycle(thread_id):
            try:
                rng = np.random.default_rng(thread_id)
                for i in range(100):
                    wid = int(rng.integers(0, 4))
                    x = rng.normal(size=4)
                    did = f"t{thread_id}-d{i}"
                    pending.add(did, {"wid": wid, "x": x, "start_ts": time.time()})
                    # Simulate some work
                    latency = rng.exponential(100)
                    metric, pt = LatencyTracker.latency_metric(latency, 50)
                    baseline = tracker.get_baseline(wid, "M", "L", pt, fallback=100.0)
                    reward = LatencyTracker.compute_reward(metric, baseline, True)
                    rec = pending.pop(did)
                    if rec:
                        beta.update(wid, reward)
                        lints.update(wid, x, reward)
                        tracker.update_baselines(wid, "M", "L", metric, pt)
                        with lock:
                            completed["count"] += 1
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(decision_feedback_cycle, i) for i in range(8)]
            for f in futures:
                f.result()

        assert not errors
        assert completed["count"] == 800
        # All workers should have been updated
        for wid in range(4):
            alpha, _ = beta.get_params(wid)
            assert alpha > 1.0  # at least some updates


# ===========================================================================
# Parameter Tolerance Tests
# ===========================================================================
class TestParameterTolerances:
    """Verify acceptable ranges and numerical properties of all learner parameters."""

    # --- BetaLearner tolerances ---
    def test_beta_alpha_beta_stay_positive(self):
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(1000):
            bl.update(1, reward=0.0)
        alpha, beta = bl.get_params(1)
        assert alpha > 0
        assert beta > 0

    def test_beta_doesnt_overflow_after_many_updates(self):
        bl = BetaLearner()
        bl.add_worker(1)
        for _ in range(100_000):
            bl.update(1, reward=0.5)
        alpha, beta = bl.get_params(1)
        assert math.isfinite(alpha) and math.isfinite(beta)
        s = bl.sample(1)
        assert math.isfinite(s) and 0 <= s <= 1

    # --- LinTSLearner tolerances ---
    def test_lints_eigenvalues_bounded(self):
        lts = LinTSLearner(feature_dim=9, lambda_=1.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(500):
            x = rng.normal(size=9)
            lts.update(1, x, reward=rng.random())
        A, _ = lts.get_params(1)
        eigs = np.linalg.eigvalsh(A)
        assert np.all(eigs > 0), f"Non-positive eigenvalue: {eigs.min()}"
        assert np.all(np.isfinite(eigs))

    def test_lints_posterior_mean_bounded(self):
        """Posterior mean shouldn't explode even with extreme inputs."""
        lts = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(200):
            x = rng.normal(size=4) * 10.0
            lts.update(1, x, reward=rng.random())
        theta = lts.posterior_mean(1)
        assert np.all(np.isfinite(theta))
        assert np.linalg.norm(theta) < 100.0

    def test_lints_condition_number_acceptable(self):
        """A matrix shouldn't become too ill-conditioned."""
        lts = LinTSLearner(feature_dim=9, lambda_=1.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        rng = np.random.default_rng(42)
        for _ in range(500):
            x = rng.normal(size=9)
            lts.update(1, x, reward=rng.random())
        A, _ = lts.get_params(1)
        cond = np.linalg.cond(A)
        assert cond < 1e10, f"Condition number too high: {cond}"

    def test_lints_forget_rate_bounds(self):
        """Forget rate should be clamped to valid range."""
        lts = LinTSLearner(feature_dim=4, forget_rate=1.5)  # out of range
        assert lts.forget_rate < 1.0
        lts2 = LinTSLearner(feature_dim=4, forget_rate=-1.0)  # out of range
        assert lts2.forget_rate > 0.0

    # --- LatencyTracker tolerances ---
    def test_reward_near_zero_baseline(self):
        """Baseline near zero should not cause division by zero."""
        r = LatencyTracker.compute_reward(metric=100.0, baseline=0.0001, success=True)
        assert math.isfinite(r)
        assert 0.0 <= r <= 1.0

    def test_reward_zero_metric(self):
        """Zero metric (instant response) should give reward near 1."""
        r = LatencyTracker.compute_reward(metric=0.0, baseline=100.0, success=True)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_ema_converges_to_constant(self):
        """EMA should converge to the constant input value."""
        lt = LatencyTracker(ema_alpha=0.2)
        for _ in range(100):
            lt.update_baselines(1, "M", "L", 42.0, True)
        val = lt.get_baseline(1, "M", "L", True, fallback=0.0)
        assert val == pytest.approx(42.0, abs=0.1)

    # --- PendingDecisions tolerances ---
    def test_pending_handles_stale_entries(self):
        """Very old decisions should all be swept."""
        pd = PendingDecisions(timeout_seconds=10.0, sweep_interval_seconds=0.0)
        now = time.time()
        for i in range(100):
            pd.add(f"d{i}", {"wid": i % 4, "start_ts": now - 100.0, "x": np.zeros(4)})
        assert pd.count() == 100
        expired = pd.sweep(now)
        assert len(expired) == 100
        assert pd.count() == 0

    def test_pending_preserves_fresh_entries(self):
        pd = PendingDecisions(timeout_seconds=120.0, sweep_interval_seconds=0.0)
        now = time.time()
        for i in range(50):
            pd.add(f"d{i}", {"wid": 1, "start_ts": now, "x": np.zeros(4)})
        expired = pd.sweep(now)
        assert len(expired) == 0
        assert pd.count() == 50
