"""
tests/test_inference_validation.py
===================================
Unit tests for ClinicalNoteInput — the Pydantic input validator
in src/inference.py (R-005).

These tests mock all heavy dependencies (torch, transformers, graph_reranker)
so they run without GPU, model weights, or the full project environment.
Only the Pydantic validation logic is tested here.

Run with:
    uv run pytest tests/test_inference_validation.py -v
"""

import warnings
import pytest
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing inference.py
# This allows tests to run without torch, transformers, or model weights.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    """Mock torch, transformers, and graph_reranker for all tests."""
    mocks = {
        "torch": MagicMock(),
        "transformers": MagicMock(),
        "src.graph_reranker": MagicMock(),
        "src.preprocessing": MagicMock(),
    }
    for module_name, mock in mocks.items():
        monkeypatch.setitem(sys.modules, module_name, mock)

    # torch.device needs to return something usable
    mocks["torch"].device = lambda x: x
    mocks["torch"].backends.mps.is_available.return_value = False
    mocks["torch"].cuda.is_available.return_value = False


# ---------------------------------------------------------------------------
# Import after mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def ClinicalNoteInput():
    from src.inference import ClinicalNoteInput
    return ClinicalNoteInput


# ---------------------------------------------------------------------------
# Empty and whitespace validation
# ---------------------------------------------------------------------------

class TestEmptyNoteValidation:

    def test_empty_string_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ClinicalNoteInput(note="")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalNoteInput(note="   ")

    def test_newlines_only_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalNoteInput(note="\n\n\n")

    def test_tabs_only_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalNoteInput(note="\t\t\t")

    def test_none_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalNoteInput(note=None)

    def test_integer_raises(self, ClinicalNoteInput):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClinicalNoteInput(note=42)


# ---------------------------------------------------------------------------
# Short note warning
# ---------------------------------------------------------------------------

class TestShortNoteWarning:

    def test_short_note_issues_warning(self, ClinicalNoteInput):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note="Patient has chest pain.")
        assert len(w) == 1
        assert "short" in str(w[0].message).lower()
        assert issubclass(w[0].category, UserWarning)

    def test_short_note_still_returns_instance(self, ClinicalNoteInput):
        """Short note warns but does not raise — still usable."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note="Patient has pain.")
        assert note.note == "Patient has pain."

    def test_exactly_20_words_no_warning(self, ClinicalNoteInput):
        """Exactly 20 words should not trigger the short note warning."""
        text = " ".join(["word"] * 20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note=text)
        short_warnings = [x for x in w if "short" in str(x.message).lower()]
        assert len(short_warnings) == 0

    def test_19_words_triggers_warning(self, ClinicalNoteInput):
        text = " ".join(["word"] * 19)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note=text)
        short_warnings = [x for x in w if "short" in str(x.message).lower()]
        assert len(short_warnings) == 1


# ---------------------------------------------------------------------------
# Long note warning
# ---------------------------------------------------------------------------

class TestLongNoteWarning:

    def test_long_note_issues_warning(self, ClinicalNoteInput):
        text = " ".join(["word"] * 401)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note=text)
        long_warnings = [x for x in w if "long" in str(x.message).lower()
                         or "truncat" in str(x.message).lower()]
        assert len(long_warnings) == 1
        assert issubclass(w[0].category, UserWarning)

    def test_exactly_400_words_no_warning(self, ClinicalNoteInput):
        text = " ".join(["word"] * 400)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note=text)
        long_warnings = [x for x in w if "truncat" in str(x.message).lower()]
        assert len(long_warnings) == 0

    def test_401_words_triggers_warning(self, ClinicalNoteInput):
        text = " ".join(["word"] * 401)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalNoteInput(note=text)
        long_warnings = [x for x in w if "truncat" in str(x.message).lower()]
        assert len(long_warnings) == 1


# ---------------------------------------------------------------------------
# Valid notes
# ---------------------------------------------------------------------------

class TestValidNotes:

    def test_typical_apso_note_accepted(self, ClinicalNoteInput):
        note_text = (
            "Assessment: Type 2 diabetes mellitus with hyperglycaemia. "
            "Plan: Adjust metformin dosage, HbA1c recheck in 3 months. "
            "Subjective: Patient reports increased thirst and frequent urination. "
            "Objective: Fasting glucose 14.2 mmol/L, BMI 31."
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note=note_text)
        assert note.note == note_text
        assert len(w) == 0

    def test_note_is_stripped(self, ClinicalNoteInput):
        """Leading/trailing whitespace is stripped from valid notes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note="  " + "word " * 25 + "  ")
        assert not note.note.startswith(" ")
        assert not note.note.endswith(" ")

    def test_preprocessed_flag_default_false(self, ClinicalNoteInput):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note="word " * 25)
        assert note.preprocessed is False

    def test_preprocessed_flag_can_be_set(self, ClinicalNoteInput):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note="word " * 25, preprocessed=True)
        assert note.preprocessed is True

    def test_unicode_content_accepted(self, ClinicalNoteInput):
        """Clinical notes may contain unicode characters."""
        text = "Assessment: Données cliniques — patient présente une douleur. " * 3
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note=text)
        assert "Données" in note.note

    def test_bytes_input_decoded_with_warning(self, ClinicalNoteInput):
        """Bytes input is decoded and a warning is issued."""
        text_bytes = ("word " * 25).encode("utf-8")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note=text_bytes)
        assert isinstance(note.note, str)
        byte_warnings = [x for x in w if "bytes" in str(x.message).lower()]
        assert len(byte_warnings) == 1


# ---------------------------------------------------------------------------
# Backward compatibility — plain str in HierarchicalPredictor.predict()
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_plain_string_creates_valid_input(self, ClinicalNoteInput):
        """
        HierarchicalPredictor.predict() wraps plain str in ClinicalNoteInput.
        Verify that wrapping a plain str produces the same result as
        constructing ClinicalNoteInput directly.
        """
        note_text = "Assessment: Lyme disease. Plan: Doxycycline 100mg. " * 3

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            direct = ClinicalNoteInput(note=note_text)
            wrapped = ClinicalNoteInput(note=note_text, preprocessed=False)

        assert direct.note == wrapped.note
        assert direct.preprocessed == wrapped.preprocessed

    def test_clinicalnoteinput_instance_passthrough(self, ClinicalNoteInput):
        """
        Passing a ClinicalNoteInput to predict() should not double-wrap.
        Simulate what predict() does: isinstance check before wrapping.
        """
        note_text = "Assessment: Chest pain. Plan: ECG, troponin. " * 3

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            note = ClinicalNoteInput(note=note_text)

        # Simulate what HierarchicalPredictor.predict() does
        if not isinstance(note, ClinicalNoteInput):
            note = ClinicalNoteInput(note=note)

        assert note.note == note_text.strip()