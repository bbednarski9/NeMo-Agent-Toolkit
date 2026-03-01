"""
Convergence and adaptability tests for Thompson Sampling learners.

Simulates realistic routing scenarios matching the multi-domain benchmark
(8 workers, 700–2000 LLM calls per experiment, 5 domains, multi-turn
sessions) and measures:

  1. Cold-start convergence — how many observations until the learner
     correctly identifies the best worker.
  2. Change-point adaptation — after a regime shift (fast worker becomes
     slow), how many observations until the learner corrects.
  3. Steady-state regret — once converged, how often is the best worker
     chosen vs a random baseline.
  4. LinTS feature learning — does the contextual bandit learn which
     features predict fast responses.
  5. Decay parameter sensitivity — convergence speed vs stability
     tradeoff across different beta_decay values.
  6. Baseline normalization — characterizes how the per-worker EMA baseline
     in LatencyTracker interacts with the reward signal.

Run with:  pytest test_convergence.py -v
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from learners import BetaLearner, LatencyTracker, LinTSLearner

# ---------------------------------------------------------------------------
# Constants matching the real deployment
# ---------------------------------------------------------------------------
NUM_WORKERS = 8
CALLS_PER_EXPERIMENT_LOW = 700
CALLS_PER_EXPERIMENT_HIGH = 2000
CALLS_PER_WORKER_LOW = CALLS_PER_EXPERIMENT_LOW // NUM_WORKERS   # ~87
CALLS_PER_WORKER_HIGH = CALLS_PER_EXPERIMENT_HIGH // NUM_WORKERS  # ~250


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceResult:
    """Summary of a convergence simulation."""
    total_observations: int
    observations_to_converge: int | None  # None if never converged
    final_best_worker_prob: float
    final_best_worker_mean: float
    fraction_correct_choices: float


def simulate_beta_direct_rewards(
    n_workers: int,
    n_obs: int,
    true_reward_rates: dict[int, float],
    decay: float = 1.0,
    convergence_threshold: float = 0.6,
    window: int = 50,
    seed: int = 42,
) -> ConvergenceResult:
    """Run a simulated bandit with direct Bernoulli rewards (no latency tracker).

    At each step: sample all workers → pick highest → observe Bernoulli(true_rate)
    reward → update.  This isolates the Beta learner from the baseline issue.
    """
    rng = np.random.default_rng(seed)
    bl = BetaLearner(decay=decay)
    for wid in range(n_workers):
        bl.add_worker(wid)

    best_wid = max(true_reward_rates, key=true_reward_rates.get)
    choices = []
    converged_at = None

    for step in range(n_obs):
        samples = {wid: bl.sample(wid) for wid in range(n_workers)}
        chosen = max(samples, key=samples.get)
        choices.append(chosen)

        reward = 1.0 if rng.random() < true_reward_rates[chosen] else 0.0
        bl.update(chosen, reward)

        if converged_at is None and len(choices) >= window:
            frac = sum(1 for c in choices[-window:] if c == best_wid) / window
            if frac >= convergence_threshold:
                converged_at = step + 1

    final_frac = sum(1 for c in choices[-window:] if c == best_wid) / window
    return ConvergenceResult(
        total_observations=n_obs,
        observations_to_converge=converged_at,
        final_best_worker_prob=final_frac,
        final_best_worker_mean=bl.mean(best_wid),
        fraction_correct_choices=sum(1 for c in choices if c == best_wid) / len(choices),
    )


def simulate_latency_based_routing(
    n_workers: int,
    n_obs: int,
    true_latencies_ms: dict[int, float],
    decay: float = 1.0,
    use_global_baseline_only: bool = False,
    convergence_threshold: float = 0.5,
    window: int = 50,
    seed: int = 42,
) -> ConvergenceResult:
    """Simulate routing with LatencyTracker reward computation.

    When ``use_global_baseline_only=True``, reward is computed against
    the global EMA only (not per-worker), which lets faster workers
    consistently score higher.
    """
    rng = np.random.default_rng(seed)
    bl = BetaLearner(decay=decay)
    tracker = LatencyTracker(ema_alpha=0.2)
    for wid in range(n_workers):
        bl.add_worker(wid)

    best_wid = min(true_latencies_ms, key=true_latencies_ms.get)  # lowest latency is best
    choices = []
    converged_at = None

    for step in range(n_obs):
        samples = {wid: bl.sample(wid) for wid in range(n_workers)}
        chosen = max(samples, key=samples.get)
        choices.append(chosen)

        latency = rng.exponential(true_latencies_ms[chosen])
        tokens_out = 50
        metric, per_tok = LatencyTracker.latency_metric(latency, tokens_out)

        if use_global_baseline_only:
            baseline = tracker._global[per_tok]
            if baseline is None:
                baseline = max(1.0, metric)
        else:
            baseline = tracker.get_baseline(chosen, "M", "L", per_tok, fallback=metric)

        reward = LatencyTracker.compute_reward(metric, baseline, True)
        bl.update(chosen, reward)
        tracker.update_baselines(chosen, "M", "L", metric, per_tok)

        if converged_at is None and len(choices) >= window:
            frac = sum(1 for c in choices[-window:] if c == best_wid) / window
            if frac >= convergence_threshold:
                converged_at = step + 1

    final_frac = sum(1 for c in choices[-window:] if c == best_wid) / window
    return ConvergenceResult(
        total_observations=n_obs,
        observations_to_converge=converged_at,
        final_best_worker_prob=final_frac,
        final_best_worker_mean=bl.mean(best_wid),
        fraction_correct_choices=sum(1 for c in choices if c == best_wid) / len(choices),
    )


def simulate_change_point_direct(
    n_workers: int,
    n_obs_before: int,
    n_obs_after: int,
    rates_before: dict[int, float],
    rates_after: dict[int, float],
    decay: float = 1.0,
    adaptation_threshold: float = 0.5,
    window: int = 30,
    seed: int = 42,
) -> tuple[int | None, float]:
    """Direct-reward change-point simulation.

    Returns: (observations_to_adapt, final_correct_fraction)
    """
    rng = np.random.default_rng(seed)
    bl = BetaLearner(decay=decay)
    for wid in range(n_workers):
        bl.add_worker(wid)

    new_best = max(rates_after, key=rates_after.get)

    # Phase 1
    for _ in range(n_obs_before):
        samples = {wid: bl.sample(wid) for wid in range(n_workers)}
        chosen = max(samples, key=samples.get)
        reward = 1.0 if rng.random() < rates_before[chosen] else 0.0
        bl.update(chosen, reward)

    # Phase 2
    choices_after: list[int] = []
    adapted_at = None
    for step in range(n_obs_after):
        samples = {wid: bl.sample(wid) for wid in range(n_workers)}
        chosen = max(samples, key=samples.get)
        choices_after.append(chosen)
        reward = 1.0 if rng.random() < rates_after[chosen] else 0.0
        bl.update(chosen, reward)

        if adapted_at is None and len(choices_after) >= window:
            frac = sum(1 for c in choices_after[-window:] if c == new_best) / window
            if frac >= adaptation_threshold:
                adapted_at = step + 1

    final_frac = (
        sum(1 for c in choices_after[-window:] if c == new_best) / window
        if len(choices_after) >= window else 0.0
    )
    return adapted_at, final_frac


# ===========================================================================
# Cold-Start Convergence Tests (Direct Rewards — isolates Beta learner)
# ===========================================================================
class TestColdStartConvergence:
    """Measure how fast the Beta learner identifies the best worker from scratch."""

    @pytest.fixture
    def clear_gap_rates(self):
        """One clearly best worker (0.8), rest mediocre (0.3)."""
        return {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}

    @pytest.fixture
    def subtle_gap_rates(self):
        """Best worker only slightly better: 0.55 vs 0.35."""
        return {i: 0.35 for i in range(NUM_WORKERS)} | {0: 0.55}

    def test_clear_gap_converges_within_budget(self, clear_gap_rates):
        """With a clear best worker, should converge well within 1 experiment budget."""
        result = simulate_beta_direct_rewards(
            n_workers=NUM_WORKERS,
            n_obs=CALLS_PER_EXPERIMENT_LOW,
            true_reward_rates=clear_gap_rates,
            decay=0.995,
        )
        assert result.observations_to_converge is not None, (
            f"Did not converge in {CALLS_PER_EXPERIMENT_LOW} obs, "
            f"final best-worker fraction: {result.final_best_worker_prob:.2f}, "
            f"mean: {result.final_best_worker_mean:.3f}"
        )
        assert result.observations_to_converge <= 300, (
            f"Converged too slowly: {result.observations_to_converge} obs"
        )

    def test_clear_gap_no_decay(self, clear_gap_rates):
        """Without decay, clear gap should also converge."""
        result = simulate_beta_direct_rewards(
            n_workers=NUM_WORKERS,
            n_obs=CALLS_PER_EXPERIMENT_LOW,
            true_reward_rates=clear_gap_rates,
            decay=1.0,
        )
        assert result.observations_to_converge is not None
        assert result.observations_to_converge <= 350

    def test_subtle_gap_converges_within_high_budget(self, subtle_gap_rates):
        """Subtle gap needs more data but should converge within the larger budget."""
        result = simulate_beta_direct_rewards(
            n_workers=NUM_WORKERS,
            n_obs=CALLS_PER_EXPERIMENT_HIGH,
            true_reward_rates=subtle_gap_rates,
            decay=0.995,
        )
        assert result.observations_to_converge is not None, (
            f"Did not converge in {CALLS_PER_EXPERIMENT_HIGH} obs, "
            f"final fraction: {result.final_best_worker_prob:.2f}"
        )

    def test_convergence_speed_scales_with_gap(self):
        """Larger reward gap → faster convergence."""
        speeds = {}
        for gap, best_rate in [(0.2, 0.5), (0.4, 0.7), (0.6, 0.9)]:
            rates = {i: best_rate - gap for i in range(NUM_WORKERS)} | {0: best_rate}
            r = simulate_beta_direct_rewards(NUM_WORKERS, 1500, rates, decay=0.995, seed=42)
            speeds[gap] = r.observations_to_converge or 1500
        # Larger gap should converge faster
        assert speeds[0.6] <= speeds[0.2]


# ===========================================================================
# Change-Point Adaptation Tests (Direct Rewards)
# ===========================================================================
class TestChangePointAdaptation:
    """After a regime shift, measure how fast the learner corrects."""

    def test_no_decay_very_slow_to_adapt(self):
        """Without decay, accumulated evidence makes adaptation very slow."""
        rates_before = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}
        rates_after = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.1, 1: 0.8}
        # Compare no-decay vs decay=0.995 — no-decay should adapt slower.
        adapted_no_decay, _ = simulate_change_point_direct(
            NUM_WORKERS, n_obs_before=500, n_obs_after=500,
            rates_before=rates_before, rates_after=rates_after, decay=1.0, seed=99,
        )
        adapted_decay, _ = simulate_change_point_direct(
            NUM_WORKERS, n_obs_before=500, n_obs_after=500,
            rates_before=rates_before, rates_after=rates_after, decay=0.995, seed=99,
        )
        no_decay_speed = adapted_no_decay if adapted_no_decay is not None else 501
        decay_speed = adapted_decay if adapted_decay is not None else 501
        assert no_decay_speed >= decay_speed, (
            f"No-decay ({no_decay_speed}) should adapt no faster than decay=0.995 ({decay_speed})"
        )

    def test_moderate_decay_adapts(self):
        """With decay=0.995 (window≈200), should adapt within a reasonable budget."""
        rates_before = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}
        rates_after = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.1, 1: 0.8}
        adapted_at, final_frac = simulate_change_point_direct(
            NUM_WORKERS, n_obs_before=500, n_obs_after=500,
            rates_before=rates_before, rates_after=rates_after, decay=0.995,
        )
        assert adapted_at is not None, (
            f"decay=0.995 did not adapt in 500 obs, final frac={final_frac:.2f}"
        )
        assert adapted_at <= 350

    def test_fast_decay_adapts_quickly(self):
        """With decay=0.990 (window≈100), should adapt fast after change."""
        rates_before = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}
        rates_after = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.1, 1: 0.8}
        adapted_at, final_frac = simulate_change_point_direct(
            NUM_WORKERS, n_obs_before=500, n_obs_after=400,
            rates_before=rates_before, rates_after=rates_after, decay=0.990,
        )
        assert adapted_at is not None, (
            f"decay=0.990 did not adapt in 400 obs, final frac={final_frac:.2f}"
        )
        assert adapted_at <= 200

    def test_gradual_degradation(self):
        """Worker gradually slows down — learner should shift to alternatives."""
        bl = BetaLearner(decay=0.995)
        rng = np.random.default_rng(42)
        for wid in range(NUM_WORKERS):
            bl.add_worker(wid)

        n_total = 1000
        choices = []
        for step in range(n_total):
            # Worker 0: degrades from 0.8 → 0.1 over the run
            # Worker 1: consistently 0.6
            rates = {i: 0.3 for i in range(NUM_WORKERS)}
            rates[0] = max(0.1, 0.8 - 0.7 * step / n_total)
            rates[1] = 0.6

            samples = {wid: bl.sample(wid) for wid in range(NUM_WORKERS)}
            chosen = max(samples, key=samples.get)
            choices.append(chosen)
            reward = 1.0 if rng.random() < rates[chosen] else 0.0
            bl.update(chosen, reward)

        # In last 200 steps, w0 rate is 0.24→0.10, w1 is 0.6.
        last_200 = choices[-200:]
        w0_frac = sum(1 for c in last_200 if c == 0) / len(last_200)
        w1_frac = sum(1 for c in last_200 if c == 1) / len(last_200)
        assert w1_frac > w0_frac, (
            f"Worker 1 ({w1_frac:.2f}) should dominate worker 0 ({w0_frac:.2f}) "
            f"by end of gradual degradation"
        )

    def test_oscillating_performance(self):
        """Workers alternate being good/bad every 150 steps.
        With decay=0.990 (window≈100), the learner should partially track."""
        bl = BetaLearner(decay=0.990)
        rng = np.random.default_rng(42)
        for wid in range(4):
            bl.add_worker(wid)

        n_total = 900
        correct = 0
        for step in range(n_total):
            phase = (step // 150) % 2
            if phase == 0:
                rates = {0: 0.8, 1: 0.2, 2: 0.2, 3: 0.2}
                best = 0
            else:
                rates = {0: 0.2, 1: 0.8, 2: 0.2, 3: 0.2}
                best = 1

            samples = {wid: bl.sample(wid) for wid in range(4)}
            chosen = max(samples, key=samples.get)
            if chosen == best:
                correct += 1
            reward = 1.0 if rng.random() < rates[chosen] else 0.0
            bl.update(chosen, reward)

        accuracy = correct / n_total
        # Random baseline = 0.25; should meaningfully beat it
        assert accuracy > 0.35, (
            f"Oscillation tracking accuracy {accuracy:.2f} (random=0.25)"
        )


# ===========================================================================
# Effective Window and Decay Sensitivity
# ===========================================================================
class TestDecaySensitivity:
    """Explore the tradeoff between convergence speed and steady-state stability."""

    @pytest.mark.parametrize("decay,expected_window", [
        (1.0, float("inf")),
        (0.998, 500),
        (0.995, 200),
        (0.990, 100),
        (0.980, 50),
    ])
    def test_effective_window_calculation(self, decay, expected_window):
        bl = BetaLearner(decay=decay)
        if math.isinf(expected_window):
            assert math.isinf(bl.effective_window)
        else:
            assert bl.effective_window == pytest.approx(expected_window, rel=0.01)

    @pytest.mark.parametrize("decay,expected_hl", [
        (0.998, 346),
        (0.995, 138),
        (0.990, 69),
        (0.980, 34),
    ])
    def test_half_life_calculation(self, decay, expected_hl):
        bl = BetaLearner(decay=decay)
        assert bl.half_life == pytest.approx(expected_hl, rel=0.02)

    def test_effective_sample_size_bounded_with_decay(self):
        """With decay, ESS should plateau around effective_window."""
        bl = BetaLearner(decay=0.995)
        bl.add_worker(1)
        ess_values = []
        for step in range(1000):
            bl.update(1, reward=0.7)
            if step % 50 == 49:
                ess_values.append(bl.effective_sample_size(1))
        assert ess_values[-1] < 300, (
            f"ESS should plateau with decay=0.995, got {ess_values[-1]:.0f}"
        )
        # Should have plateaued
        assert abs(ess_values[-1] - ess_values[-2]) / max(ess_values[-1], 1) < 0.05

    def test_no_decay_ess_grows_linearly(self):
        """Without decay, ESS grows with every observation."""
        bl = BetaLearner(decay=1.0)
        bl.add_worker(1)
        for _ in range(500):
            bl.update(1, reward=0.6)
        ess = bl.effective_sample_size(1)
        assert ess > 400

    def test_all_decay_values_converge_clear_gap(self):
        """All tested decay values should converge with a clear reward gap."""
        rates = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}
        for decay in [1.0, 0.998, 0.995, 0.992, 0.990]:
            r = simulate_beta_direct_rewards(
                NUM_WORKERS, 1500, rates, decay=decay, seed=42,
            )
            assert r.observations_to_converge is not None, (
                f"decay={decay} did not converge in 1500 obs"
            )
            assert r.final_best_worker_prob > 0.5, (
                f"decay={decay} final prob {r.final_best_worker_prob:.2f} too low"
            )


# ===========================================================================
# Latency-Based Reward Signal Analysis
# ===========================================================================
class TestLatencyRewardSignal:
    """Characterize how LatencyTracker baselines affect the reward signal.

    KEY FINDING: Per-worker baselines equalize rewards across workers,
    sabotaging the Beta learner's ability to differentiate.  Using
    global-only baselines preserves the signal.
    """

    def test_per_worker_baselines_equalize_rewards(self):
        """Per-worker EMA baselines cause all workers to converge toward reward≈0.5."""
        tracker = LatencyTracker(ema_alpha=0.2)
        rng = np.random.default_rng(42)

        # Feed 100 observations from a fast worker and a slow worker
        fast_rewards, slow_rewards = [], []
        for _ in range(100):
            # Fast worker: 100ms latency
            metric_fast, pt = LatencyTracker.latency_metric(rng.exponential(100.0), 50)
            bl_fast = tracker.get_baseline(0, "M", "L", pt, fallback=metric_fast)
            fast_rewards.append(LatencyTracker.compute_reward(metric_fast, bl_fast, True))
            tracker.update_baselines(0, "M", "L", metric_fast, pt)

            # Slow worker: 1000ms latency
            metric_slow, pt = LatencyTracker.latency_metric(rng.exponential(1000.0), 50)
            bl_slow = tracker.get_baseline(1, "M", "L", pt, fallback=metric_slow)
            slow_rewards.append(LatencyTracker.compute_reward(metric_slow, bl_slow, True))
            tracker.update_baselines(1, "M", "L", metric_slow, pt)

        # After baselines converge per-worker, rewards should cluster near 0.5
        # for both workers (this is the PROBLEM)
        fast_tail = np.mean(fast_rewards[-30:])
        slow_tail = np.mean(slow_rewards[-30:])
        gap = abs(fast_tail - slow_tail)
        assert gap < 0.15, (
            f"With per-worker baselines, expected near-equalized rewards "
            f"but gap={gap:.3f} (fast={fast_tail:.3f}, slow={slow_tail:.3f})"
        )

    def test_global_only_baseline_preserves_signal(self):
        """Using only the global baseline preserves reward differentiation."""
        tracker = LatencyTracker(ema_alpha=0.2)
        rng = np.random.default_rng(42)

        fast_rewards, slow_rewards = [], []
        for _ in range(200):
            # Fast worker
            metric_fast, pt = LatencyTracker.latency_metric(rng.exponential(100.0), 50)
            global_bl = tracker._global[pt] or max(1.0, metric_fast)
            fast_rewards.append(LatencyTracker.compute_reward(metric_fast, global_bl, True))
            tracker.update_baselines(0, "M", "L", metric_fast, pt)

            # Slow worker
            metric_slow, pt = LatencyTracker.latency_metric(rng.exponential(1000.0), 50)
            global_bl = tracker._global[pt] or max(1.0, metric_slow)
            slow_rewards.append(LatencyTracker.compute_reward(metric_slow, global_bl, True))
            tracker.update_baselines(1, "M", "L", metric_slow, pt)

        fast_tail = np.mean(fast_rewards[-50:])
        slow_tail = np.mean(slow_rewards[-50:])
        assert fast_tail > slow_tail + 0.1, (
            f"Global baseline should differentiate: fast={fast_tail:.3f} > slow={slow_tail:.3f}"
        )

    def test_global_baseline_enables_beta_convergence(self):
        """With global-only baseline, the full latency-based system converges."""
        latencies = {i: 500.0 for i in range(NUM_WORKERS)} | {0: 100.0}
        result = simulate_latency_based_routing(
            NUM_WORKERS, 1000, latencies,
            decay=0.995, use_global_baseline_only=True,
        )
        assert result.observations_to_converge is not None, (
            f"Global baseline system did not converge in 1000 obs, "
            f"final frac: {result.final_best_worker_prob:.2f}"
        )

    def test_per_worker_baseline_hinders_convergence(self):
        """With per-worker baselines, convergence is much slower or fails."""
        latencies = {i: 500.0 for i in range(NUM_WORKERS)} | {0: 100.0}
        result_per_worker = simulate_latency_based_routing(
            NUM_WORKERS, 1000, latencies,
            decay=0.995, use_global_baseline_only=False,
        )
        result_global = simulate_latency_based_routing(
            NUM_WORKERS, 1000, latencies,
            decay=0.995, use_global_baseline_only=True,
        )
        # Global should converge better than per-worker
        assert result_global.fraction_correct_choices > result_per_worker.fraction_correct_choices

    def test_get_global_baseline_api_enables_convergence(self):
        """Verify that using LatencyTracker.get_global_baseline() (the public API
        that the router now calls in 'global' mode) enables convergence."""
        bl = BetaLearner(decay=0.995)
        tracker = LatencyTracker(ema_alpha=0.2)
        rng = np.random.default_rng(42)

        for wid in range(NUM_WORKERS):
            bl.add_worker(wid)

        true_latencies = {i: 500.0 for i in range(NUM_WORKERS)}
        true_latencies[0] = 100.0
        best_wid = 0
        window = 50
        choices = []

        for step in range(1000):
            samples = {wid: bl.sample(wid) for wid in range(NUM_WORKERS)}
            chosen = max(samples, key=samples.get)
            choices.append(chosen)

            latency = rng.exponential(true_latencies[chosen])
            metric, per_tok = LatencyTracker.latency_metric(latency, 50)
            baseline = tracker.get_global_baseline(per_tok, fallback=metric)
            reward = LatencyTracker.compute_reward(metric, baseline, True)
            bl.update(chosen, reward)
            tracker.update_baselines(chosen, "M", "L", metric, per_tok)

        final_frac = sum(1 for c in choices[-window:] if c == best_wid) / window
        assert final_frac > 0.4, (
            f"get_global_baseline API: best worker chosen {final_frac:.2f} in last "
            f"{window} decisions (expected > 0.4)"
        )


# ===========================================================================
# LinTS Convergence Tests
# ===========================================================================
class TestLinTSConvergence:
    """Contextual bandit learning convergence tests."""

    def test_learns_linear_relationship(self):
        """LinTS should learn theta such that high x[1] → high score."""
        lts = LinTSLearner(feature_dim=4, lambda_=0.5, v=0.1, forget_rate=0.999)
        rng = np.random.default_rng(42)
        lts.add_worker(1)

        true_theta = np.array([0.3, 0.8, -0.2, 0.0])
        for _ in range(500):
            x = rng.normal(size=4)
            x[0] = 1.0
            true_score = 1.0 / (1.0 + np.exp(-true_theta @ x))
            reward = 1.0 if rng.random() < true_score else 0.0
            lts.update(1, x, reward)

        learned = lts.posterior_mean(1)
        assert learned[1] > 0.1, f"theta[1]={learned[1]:.3f}, expected positive"
        assert learned[2] < 0.0, f"theta[2]={learned[2]:.3f}, expected negative"

    def test_cold_start_stabilizes(self):
        """Posterior mean should stabilize within 400 observations."""
        lts = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        rng = np.random.default_rng(42)
        lts.add_worker(1)

        true_theta = np.array([0.5, 0.6, 0.0, 0.0])
        means = []
        for step in range(400):
            x = rng.normal(size=4)
            x[0] = 1.0
            true_score = 1.0 / (1.0 + np.exp(-true_theta @ x))
            reward = min(1.0, max(0.0, true_score + rng.normal(0, 0.1)))
            lts.update(1, x, reward)
            if step % 20 == 19:
                means.append(lts.posterior_mean(1).copy())

        diffs = [np.linalg.norm(means[-1] - means[i]) for i in range(-3, -1)]
        assert all(d < 0.3 for d in diffs), (
            f"Posterior mean still shifting after 400 obs: diffs={diffs}"
        )

    def test_forget_rate_enables_feature_adaptation(self):
        """After regime change in features, LinTS with forget_rate adapts."""
        lts = LinTSLearner(feature_dim=3, lambda_=0.5, v=0.1, forget_rate=0.990)
        rng = np.random.default_rng(42)
        lts.add_worker(1)

        # Phase 1: x[1] predicts reward
        for _ in range(300):
            x = np.array([1.0, rng.random(), rng.random()])
            reward = min(1.0, max(0.0, 0.5 + 0.4 * x[1] + rng.normal(0, 0.05)))
            lts.update(1, x, reward)
        theta_p1 = lts.posterior_mean(1).copy()

        # Phase 2: x[2] predicts reward instead
        for _ in range(300):
            x = np.array([1.0, rng.random(), rng.random()])
            reward = min(1.0, max(0.0, 0.5 + 0.4 * x[2] + rng.normal(0, 0.05)))
            lts.update(1, x, reward)
        theta_p2 = lts.posterior_mean(1)

        assert theta_p2[2] > theta_p1[2] + 0.05, (
            f"theta[2] should increase: before={theta_p1[2]:.3f}, after={theta_p2[2]:.3f}"
        )

    def test_worker_differentiation(self):
        """Workers with different reward histories should get different scores."""
        lts = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.1, forget_rate=0.995)
        for wid in range(4):
            lts.add_worker(wid)

        x = np.array([1.0, 0.5, 0.3, 0.1])
        for _ in range(200):
            lts.update(0, x, reward=0.9)
            lts.update(1, x, reward=0.1)

        x_test = np.array([1.0, 0.5, 0.3, 0.1])
        scores_0 = [lts.sample(0, x_test) for _ in range(200)]
        scores_1 = [lts.sample(1, x_test) for _ in range(200)]
        assert np.mean(scores_0) > np.mean(scores_1)


# ===========================================================================
# Combined Learner System Tests
# ===========================================================================
class TestCombinedLearnerSystem:
    """Test beta + lints + latency tracker working together."""

    def test_system_with_global_baseline_converges(self):
        """Full routing system converges when using global baseline for rewards."""
        beta = BetaLearner(decay=0.995)
        lints = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        tracker = LatencyTracker(ema_alpha=0.2)
        rng = np.random.default_rng(42)

        for wid in range(NUM_WORKERS):
            beta.add_worker(wid)
            lints.add_worker(wid)

        true_latencies = {i: 500.0 for i in range(NUM_WORKERS)}
        true_latencies[0] = 100.0  # Fast worker

        choices = []
        x = np.array([1.0, 0.5, 0.3, 0.1])
        for step in range(CALLS_PER_EXPERIMENT_HIGH):
            scores = {}
            for wid in range(NUM_WORKERS):
                scores[wid] = 0.05 * beta.sample(wid) + math.tanh(lints.sample(wid, x))
            chosen = max(scores, key=scores.get)
            choices.append(chosen)

            latency = rng.exponential(true_latencies[chosen])
            metric, pt = LatencyTracker.latency_metric(latency, 50)
            # Use global baseline
            baseline = tracker._global[pt] or max(1.0, metric)
            reward = LatencyTracker.compute_reward(metric, baseline, True)
            beta.update(chosen, reward)
            lints.update(chosen, x, reward)
            tracker.update_baselines(chosen, "M", "L", metric, pt)

        second_half = choices[len(choices) // 2:]
        w0_frac = sum(1 for c in second_half if c == 0) / len(second_half)
        assert w0_frac > 0.25, (
            f"Worker 0 (fastest) chosen only {w0_frac:.2f} in second half"
        )

    def test_system_adapts_after_worker_failure(self):
        """When best worker fails, system reroutes using direct rewards."""
        beta = BetaLearner(decay=0.995)
        rng = np.random.default_rng(42)
        n_workers = 4
        for wid in range(n_workers):
            beta.add_worker(wid)

        # Phase 1: worker 0 is best
        for _ in range(300):
            rates = {0: 0.8, 1: 0.3, 2: 0.3, 3: 0.3}
            samples = {wid: beta.sample(wid) for wid in range(n_workers)}
            chosen = max(samples, key=samples.get)
            reward = 1.0 if rng.random() < rates[chosen] else 0.0
            beta.update(chosen, reward)

        # Phase 2: worker 0 fails, worker 2 becomes best
        choices_p2 = []
        for _ in range(400):
            rates = {0: 0.05, 1: 0.3, 2: 0.8, 3: 0.3}
            samples = {wid: beta.sample(wid) for wid in range(n_workers)}
            chosen = max(samples, key=samples.get)
            choices_p2.append(chosen)
            reward = 1.0 if rng.random() < rates[chosen] else 0.0
            beta.update(chosen, reward)

        last_100 = choices_p2[-100:]
        w0_frac = sum(1 for c in last_100 if c == 0) / len(last_100)
        w2_frac = sum(1 for c in last_100 if c == 2) / len(last_100)
        assert w0_frac < 0.15, f"Failed worker 0 still chosen {w0_frac:.2f}"
        assert w2_frac > 0.3, f"New best worker 2 only chosen {w2_frac:.2f}"

    def test_recommended_decay_for_dataset_size(self):
        """Verify decay=0.995 is appropriate for our 700–2000 call experiments."""
        bl = BetaLearner(decay=0.995)
        assert 150 <= bl.effective_window <= 250
        assert 100 <= bl.half_life <= 200

        rates = {i: 0.3 for i in range(NUM_WORKERS)} | {0: 0.8}
        result = simulate_beta_direct_rewards(
            NUM_WORKERS, CALLS_PER_EXPERIMENT_LOW, rates, decay=0.995,
        )
        assert result.observations_to_converge is not None
        assert result.fraction_correct_choices > 0.25
