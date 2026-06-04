"""
tests/test_train_seeding.py
===========================
Tests for _set_all_seeds in scripts/train.py.

Before this, train.py seeded only the data split; model init, dropout, and
batch shuffling drew from unseeded global RNG, so runs were not reproducible.
These tests assert that _set_all_seeds actually pins python/numpy/torch RNG so
that subsequent random draws are reproducible across calls with the same seed
(and differ across different seeds).
"""
import sys
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_set_all_seeds():
    """Import _set_all_seeds from scripts/train.py without running main()."""
    spec = importlib.util.spec_from_file_location("train_mod", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._set_all_seeds


class TestSetAllSeeds:
    def test_python_random_reproducible(self):
        import random
        set_seeds = _load_set_all_seeds()
        set_seeds(42)
        a = [random.random() for _ in range(5)]
        set_seeds(42)
        b = [random.random() for _ in range(5)]
        assert a == b, "python random not reproducible after _set_all_seeds"

    def test_numpy_reproducible(self):
        np = pytest.importorskip("numpy")
        set_seeds = _load_set_all_seeds()
        set_seeds(123)
        a = np.random.rand(5).tolist()
        set_seeds(123)
        b = np.random.rand(5).tolist()
        assert a == b, "numpy RNG not reproducible after _set_all_seeds"

    def test_torch_reproducible(self):
        torch = pytest.importorskip("torch")
        set_seeds = _load_set_all_seeds()
        set_seeds(7)
        a = torch.rand(5).tolist()
        set_seeds(7)
        b = torch.rand(5).tolist()
        assert a == b, "torch RNG not reproducible after _set_all_seeds"

    def test_different_seeds_differ(self):
        import random
        set_seeds = _load_set_all_seeds()
        set_seeds(1)
        a = [random.random() for _ in range(5)]
        set_seeds(2)
        b = [random.random() for _ in range(5)]
        assert a != b, "different seeds produced identical draws (seed not applied)"

    def test_does_not_raise_without_optional_libs(self):
        """_set_all_seeds must tolerate numpy/torch absence (wrapped in try)."""
        set_seeds = _load_set_all_seeds()
        # Should not raise regardless of what's installed.
        set_seeds(0)
