"""
Unit tests for learner serialization: to_dict, from_dict, load_state, reset_all, reset.

Tests cover:
  - BetaLearner:  to_dict/from_dict/load_state/reset_all structure and roundtrip
  - LinTSLearner: to_dict/from_dict/load_state/reset_all structure and roundtrip
  - LatencyTracker: reset clears global, worker, and bucket baselines

Run with:  pytest test_learner_serialization.py -v
"""

import numpy as np
import pytest

from learners import BetaLearner, LatencyTracker, LinTSLearner


# ===========================================================================
# TestBetaLearnerSerialization
# ===========================================================================
class TestBetaLearnerSerialization:
    """Tests for BetaLearner serialization and reset methods."""

    def test_to_dict_structure(self):
        """Verify the dict has keys: type, decay, min_pseudo_count, bandits."""
        bl = BetaLearner(decay=0.995, min_pseudo_count=0.5)
        bl.add_worker(1, alpha=2.0, beta=3.0)
        d = bl.to_dict()
        assert set(d.keys()) == {"type", "decay", "min_pseudo_count", "bandits"}
        assert d["type"] == "BetaLearner"
        assert d["decay"] == 0.995
        assert d["min_pseudo_count"] == 0.5
        assert "1" in d["bandits"]
        assert d["bandits"]["1"] == [2.0, 3.0]

    def test_roundtrip_preserves_state(self):
        """Add workers, do updates, to_dict then from_dict, verify params match."""
        bl = BetaLearner(decay=0.99, min_pseudo_count=1.0)
        bl.add_worker(1)
        bl.add_worker(2)
        for _ in range(20):
            bl.update(1, reward=0.8)
            bl.update(2, reward=0.2)

        d = bl.to_dict()
        restored = BetaLearner.from_dict(d)

        assert restored.decay == bl.decay
        assert restored.min_pseudo_count == bl.min_pseudo_count
        assert restored.worker_ids == bl.worker_ids
        for wid in [1, 2]:
            a_orig, b_orig = bl.get_params(wid)
            a_restored, b_restored = restored.get_params(wid)
            assert a_restored == pytest.approx(a_orig)
            assert b_restored == pytest.approx(b_orig)

    def test_from_dict_restores_bandits(self):
        """Create from_dict with known values, verify get_params returns them."""
        data = {
            "type": "BetaLearner",
            "decay": 0.98,
            "min_pseudo_count": 0.6,
            "bandits": {"1": [5.0, 3.0], "2": [2.0, 8.0]},
        }
        bl = BetaLearner.from_dict(data)
        assert bl.get_params(1) == (5.0, 3.0)
        assert bl.get_params(2) == (2.0, 8.0)
        assert bl.decay == 0.98
        assert bl.min_pseudo_count == 0.6

    def test_load_state_in_place(self):
        """Create learner, load_state with different values, verify in-place mutation."""
        bl = BetaLearner(decay=1.0, min_pseudo_count=1.0)
        bl.add_worker(1, alpha=10.0, beta=10.0)
        obj_id = id(bl)

        new_state = {
            "decay": 0.95,
            "min_pseudo_count": 0.8,
            "bandits": {"1": [1.5, 2.5], "2": [3.0, 4.0]},
        }
        bl.load_state(new_state)

        assert id(bl) == obj_id
        assert bl.decay == 0.95
        assert bl.min_pseudo_count == 0.8
        assert bl.get_params(1) == (1.5, 2.5)
        assert bl.get_params(2) == (3.0, 4.0)
        assert set(bl.worker_ids) == {1, 2}

    def test_reset_all(self):
        """Add workers, do updates so alpha/beta diverge from 1.0, call reset_all, verify all workers back to (1.0, 1.0)."""
        bl = BetaLearner()
        bl.add_worker(1)
        bl.add_worker(2)
        for _ in range(50):
            bl.update(1, reward=0.9)
            bl.update(2, reward=0.1)

        bl.reset_all()

        for wid in [1, 2]:
            alpha, beta = bl.get_params(wid)
            assert alpha == pytest.approx(1.0)
            assert beta == pytest.approx(1.0)

    def test_reset_all_preserves_worker_ids(self):
        """After reset_all, worker_ids should be unchanged."""
        bl = BetaLearner()
        bl.add_worker(1)
        bl.add_worker(2)
        bl.add_worker(3)
        for _ in range(10):
            bl.update(1, reward=0.5)

        ids_before = set(bl.worker_ids)
        bl.reset_all()
        ids_after = set(bl.worker_ids)
        assert ids_after == ids_before
        assert ids_after == {1, 2, 3}


# ===========================================================================
# TestLinTSLearnerSerialization
# ===========================================================================
class TestLinTSLearnerSerialization:
    """Tests for LinTSLearner serialization and reset methods."""

    def test_to_dict_structure(self):
        """Verify keys: type, feature_dim, lambda, v, forget_rate, workers (with A and b)."""
        lts = LinTSLearner(feature_dim=4, lambda_=2.0, v=0.3, forget_rate=0.99)
        lts.add_worker(1)
        d = lts.to_dict()

        assert set(d.keys()) == {"type", "feature_dim", "lambda", "v", "forget_rate", "workers"}
        assert d["type"] == "LinTSLearner"
        assert d["feature_dim"] == 4
        assert d["lambda"] == 2.0
        assert d["v"] == 0.3
        assert d["forget_rate"] == 0.99
        assert "1" in d["workers"]
        assert "A" in d["workers"]["1"]
        assert "b" in d["workers"]["1"]
        assert len(d["workers"]["1"]["A"]) == 4
        assert len(d["workers"]["1"]["b"]) == 4

    def test_roundtrip_preserves_state(self):
        """Add workers, do updates, to_dict then from_dict, verify A and b match (use np.testing.assert_allclose)."""
        lts = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        lts.add_worker(2)
        x1 = np.array([1.0, 0.5, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.5, 0.2])
        for _ in range(20):
            lts.update(1, x1, reward=0.8)
            lts.update(2, x2, reward=0.3)

        d = lts.to_dict()
        restored = LinTSLearner.from_dict(d)

        assert restored.feature_dim == lts.feature_dim
        assert restored.lambda_ == lts.lambda_
        assert restored.v == lts.v
        assert restored.forget_rate == lts.forget_rate
        assert restored.worker_ids == lts.worker_ids
        for wid in [1, 2]:
            A_orig, b_orig = lts.get_params(wid)
            A_restored, b_restored = restored.get_params(wid)
            np.testing.assert_allclose(A_restored, A_orig)
            np.testing.assert_allclose(b_restored, b_orig)

    def test_from_dict_restores_matrices(self):
        """Create from_dict with known A/b, verify get_params."""
        lambda_val = 1.0
        A1 = [[2.0, 0.5, 0.0, 0.0], [0.5, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        b1 = [1.0, 0.5, 0.0, 0.0]
        data = {
            "type": "LinTSLearner",
            "feature_dim": 4,
            "lambda": lambda_val,
            "v": 0.25,
            "forget_rate": 0.995,
            "workers": {"1": {"A": A1, "b": b1}},
        }
        lts = LinTSLearner.from_dict(data)
        A, b = lts.get_params(1)
        np.testing.assert_allclose(A, np.array(A1))
        np.testing.assert_allclose(b, np.array(b1))

    def test_load_state_in_place(self):
        """Create learner, load_state, verify in-place mutation."""
        lts = LinTSLearner(feature_dim=4, lambda_=1.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        lts.update(1, x, reward=0.5)
        obj_id = id(lts)

        new_A = [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 3.0]]
        new_b = [2.0, 1.0, 0.5, 0.0]
        new_state = {
            "v": 0.5,
            "forget_rate": 0.99,
            "workers": {"1": {"A": new_A, "b": new_b}, "2": {"A": new_A, "b": [0.0, 0.0, 0.0, 0.0]}},
        }
        lts.load_state(new_state)

        assert id(lts) == obj_id
        assert lts.v == 0.5
        assert lts.forget_rate == 0.99
        A1, b1 = lts.get_params(1)
        np.testing.assert_allclose(A1, np.array(new_A))
        np.testing.assert_allclose(b1, np.array(new_b))
        assert set(lts.worker_ids) == {1, 2}

    def test_reset_all(self):
        """Do updates, call reset_all, verify all workers have A=lambda*I and b=zeros."""
        lts = LinTSLearner(feature_dim=4, lambda_=2.0, v=0.25, forget_rate=0.995)
        lts.add_worker(1)
        lts.add_worker(2)
        x = np.array([1.0, 0.5, 0.3, 0.1])
        for _ in range(30):
            lts.update(1, x, reward=0.7)
            lts.update(2, x, reward=0.3)

        lts.reset_all()

        lambda_I = 2.0 * np.eye(4)
        for wid in [1, 2]:
            A, b = lts.get_params(wid)
            np.testing.assert_allclose(A, lambda_I)
            np.testing.assert_allclose(b, np.zeros(4))

    def test_reset_all_preserves_worker_ids(self):
        """After reset_all, worker_ids should be unchanged."""
        lts = LinTSLearner(feature_dim=4)
        lts.add_worker(1)
        lts.add_worker(2)
        lts.add_worker(3)
        x = np.ones(4)
        for _ in range(10):
            lts.update(1, x, reward=0.5)

        ids_before = set(lts.worker_ids)
        lts.reset_all()
        ids_after = set(lts.worker_ids)
        assert ids_after == ids_before
        assert ids_after == {1, 2, 3}


# ===========================================================================
# TestLatencyTrackerReset
# ===========================================================================
class TestLatencyTrackerReset:
    """Tests for LatencyTracker reset method."""

    def test_reset_clears_global(self):
        """Update baselines, reset, verify get_global_baseline returns fallback."""
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "M", "L", 100.0, True)
        lt.update_baselines(2, "M", "L", 200.0, False)
        assert lt.get_global_baseline(True, fallback=999.0) != 999.0
        assert lt.get_global_baseline(False, fallback=888.0) != 888.0

        lt.reset()
        assert lt.get_global_baseline(True, fallback=999.0) == pytest.approx(999.0)
        assert lt.get_global_baseline(False, fallback=888.0) == pytest.approx(888.0)

    def test_reset_clears_worker(self):
        """Update baselines for worker, reset, verify worker baseline falls through to fallback."""
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "LOW", "LOW", 50.0, True)
        # Worker 1 has per-worker baseline; different bucket should fall through to worker
        assert lt.get_baseline(1, "HIGH", "HIGH", True, fallback=999.0) == pytest.approx(50.0)

        lt.reset()
        # After reset, no worker baseline → falls through to fallback
        assert lt.get_baseline(1, "HIGH", "HIGH", True, fallback=999.0) == pytest.approx(999.0)

    def test_reset_clears_bucket(self):
        """Update baselines, reset, verify bucket baseline gone."""
        lt = LatencyTracker(ema_alpha=0.2)
        lt.update_baselines(1, "LOW", "LOW", 50.0, True)
        assert lt.get_baseline(1, "LOW", "LOW", True, fallback=999.0) == pytest.approx(50.0)

        lt.reset()
        # After reset, bucket is gone → falls through to worker (empty) → global (empty) → fallback
        assert lt.get_baseline(1, "LOW", "LOW", True, fallback=999.0) == pytest.approx(999.0)
