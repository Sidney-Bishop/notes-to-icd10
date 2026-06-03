"""
tests/test_split_logic.py
=========================
Unit tests for scripts/train.py::_split_dataframe — the deterministic
stratified 80/10/10 train/val/test split used across all hierarchical
training.

These tests assert CORRECT behavior (test-driven). Where the current
implementation may diverge from correct behavior — specifically the
question of whether the split is stable under input ROW ORDER — the test
is written to assert the property we want, so a failure documents the bug.

Run with:
    uv run pytest tests/test_split_logic.py -v
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import _split_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frame(n_per_class: int = 50, classes=("A", "B", "C", "D")) -> pl.DataFrame:
    """
    Build a synthetic gold-like frame with a stratifiable label column.
    n_per_class rows for each class so an 80/10/10 stratified split is clean.
    """
    rows = []
    for cls in classes:
        for i in range(n_per_class):
            rows.append({
                "apso_note": f"note text for {cls} sample {i}",
                "standard_icd10": f"{cls}{i:02d}",   # unique-ish code
                "chapter": cls,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------

class TestSplitProportions:
    def test_split_is_80_10_10(self):
        df = _make_frame(n_per_class=50)            # 200 rows
        train, val, test = _split_dataframe(df, label_col="chapter", seed=42)
        total = len(df)
        assert len(train) == pytest.approx(0.8 * total, abs=2)
        assert len(val)   == pytest.approx(0.1 * total, abs=2)
        assert len(test)  == pytest.approx(0.1 * total, abs=2)

    def test_split_partitions_cover_all_rows_without_overlap(self):
        df = _make_frame(n_per_class=50)
        train, val, test = _split_dataframe(df, label_col="chapter", seed=42)
        # No row lost, no row duplicated across partitions
        assert len(train) + len(val) + len(test) == len(df)
        notes_all = set(df["apso_note"].to_list())
        notes_split = (
            set(train["apso_note"].to_list())
            | set(val["apso_note"].to_list())
            | set(test["apso_note"].to_list())
        )
        assert notes_split == notes_all
        # pairwise disjoint
        assert set(train["apso_note"]).isdisjoint(set(val["apso_note"]))
        assert set(train["apso_note"]).isdisjoint(set(test["apso_note"]))
        assert set(val["apso_note"]).isdisjoint(set(test["apso_note"]))


# ---------------------------------------------------------------------------
# Determinism under fixed seed
# ---------------------------------------------------------------------------

class TestSplitDeterminism:
    def test_same_input_same_seed_is_identical(self):
        df = _make_frame(n_per_class=50)
        a = _split_dataframe(df, label_col="chapter", seed=42)
        b = _split_dataframe(df, label_col="chapter", seed=42)
        for part_a, part_b in zip(a, b):
            assert part_a["apso_note"].to_list() == part_b["apso_note"].to_list()

    def test_different_seed_changes_partition(self):
        df = _make_frame(n_per_class=50)
        a_train, _, _ = _split_dataframe(df, label_col="chapter", seed=42)
        b_train, _, _ = _split_dataframe(df, label_col="chapter", seed=7)
        assert a_train["apso_note"].to_list() != b_train["apso_note"].to_list()


# ---------------------------------------------------------------------------
# Row-order stability — PINS THE KNOWN FRAGILITY
# ---------------------------------------------------------------------------

class TestSplitRowOrderStability:
    """
    The split should depend only on (content, seed), NOT on the order rows
    happen to arrive in. If train_test_split is applied to a differently
    ordered frame and yields a different partition, then any consumer that
    regenerates splits from re-ordered gold (e.g. SupCon reusing E-016's
    Z splits) risks train/val contamination.

    This test asserts the property we WANT (order-independence). If the
    implementation is order-dependent, this test fails and documents the bug.
    """
    def test_split_is_stable_under_row_reordering(self):
        df = _make_frame(n_per_class=50)
        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=999)

        train_a, val_a, test_a = _split_dataframe(df, label_col="chapter", seed=42)
        train_b, val_b, test_b = _split_dataframe(df_shuffled, label_col="chapter", seed=42)

        # The SET of notes in each partition should be identical regardless
        # of input order (content-addressed, not position-addressed).
        assert set(test_a["apso_note"]) == set(test_b["apso_note"]), (
            "Test partition changed when input rows were reordered — split is "
            "order-dependent, which can cause train/val contamination when "
            "splits are regenerated from re-ordered gold."
        )
        assert set(val_a["apso_note"]) == set(val_b["apso_note"])
        assert set(train_a["apso_note"]) == set(train_b["apso_note"])


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

class TestSplitStratification:
    def test_all_classes_present_in_train(self):
        df = _make_frame(n_per_class=50, classes=("A", "B", "C", "D"))
        train, _, _ = _split_dataframe(df, label_col="chapter", seed=42)
        assert set(train["chapter"].unique()) == {"A", "B", "C", "D"}

    def test_class_proportions_roughly_preserved_in_test(self):
        df = _make_frame(n_per_class=50, classes=("A", "B", "C", "D"))
        _, _, test = _split_dataframe(df, label_col="chapter", seed=42)
        counts = test.group_by("chapter").len().sort("chapter")
        # each class ~equal in a balanced frame
        lens = counts["len"].to_list()
        assert max(lens) - min(lens) <= 2
