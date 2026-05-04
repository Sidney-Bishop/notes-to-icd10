"""
tests/test_hierarchical_predictor.py
=====================================
Mock-isolated unit tests for HierarchicalPredictor in src/inference.py.

Strategy: rather than threading mocks through from_pretrained → .to(device),
we let __init__ complete with dummy internals, then directly inject the state
we care about (id2chapter, stage1_tokenizer, stage1_model, etc.) onto the
predictor instance before calling predict(). This is simpler and more robust.

Run with:
    uv run pytest tests/test_hierarchical_predictor.py -v
"""

import json
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_imports(monkeypatch, tmp_path):
    # --- torch ---
    torch_mock = MagicMock()
    torch_mock.device = lambda x: x
    torch_mock.backends.mps.is_available.return_value = False
    torch_mock.cuda.is_available.return_value = False
    torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    argmax_result = MagicMock()
    argmax_result.item.return_value = 0
    torch_mock.argmax.return_value = argmax_result
    probs_array = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
    softmax_result = MagicMock()
    softmax_result.cpu.return_value.numpy.return_value = np.array([probs_array])
    torch_mock.softmax.return_value = softmax_result
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torch.backends", torch_mock.backends)
    monkeypatch.setitem(sys.modules, "numpy", np)

    # --- transformers ---
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())

    # --- graph_reranker ---
    reranker_mock = MagicMock()
    reranker_mock.GraphReranker.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "src.graph_reranker", reranker_mock)

    # --- preprocessing ---
    preprocessing_mock = MagicMock()
    preprocessing_mock.prepare_inference_input.side_effect = lambda x: x
    monkeypatch.setitem(sys.modules, "src.preprocessing", preprocessing_mock)

    # --- pydantic: real BaseModel so ClinicalNoteInput() works in predict() ---
    class _FakeBaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    pydantic_mock = MagicMock()
    pydantic_mock.BaseModel = _FakeBaseModel
    pydantic_mock.field_validator = lambda *a, **kw: (lambda f: f)
    pydantic_mock.model_validator = lambda *a, **kw: (lambda f: f)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mock)

    # --- src.paths ---
    paths_mock = MagicMock()
    exp_paths = MagicMock()
    exp_paths.stage1_model_dir.return_value = tmp_path / "stage1"
    exp_paths.stage1_temperature_existing.return_value = None
    exp_paths.stage2_base = tmp_path / "stage2"
    exp_paths.stage2_results.return_value = tmp_path / "stage2" / "stage2_results.json"
    exp_paths.stage2_model_dir.return_value = None
    exp_paths.stage2_label_map.return_value = None
    exp_paths.stage2_temperature_existing.return_value = None
    paths_mock.ExperimentPaths.return_value = exp_paths
    monkeypatch.setitem(sys.modules, "src.paths", paths_mock)

    # --- src.config ---
    config_mock = MagicMock()
    config_mock.config.project_root = tmp_path
    config_mock.config.resolve_path.return_value = tmp_path / "outputs" / "evaluations"
    monkeypatch.setitem(sys.modules, "src.config", config_mock)

    (tmp_path / "stage1").mkdir(parents=True)
    (tmp_path / "stage2").mkdir(parents=True)

    mock_imports.torch = torch_mock
    mock_imports.reranker_mod = reranker_mock
    mock_imports.exp_paths = exp_paths
    mock_imports.tmp = tmp_path
    yield mock_imports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictor(**kwargs):
    sys.modules.pop("src.inference", None)
    from src.inference import HierarchicalPredictor
    return HierarchicalPredictor(**kwargs)


def _inject_stage1(predictor, tokenizer, model, id2chapter: dict):
    """Replace predictor Stage-1 internals with controlled test doubles."""
    predictor.stage1_tokenizer = tokenizer
    predictor.stage1_model = model
    predictor.id2chapter = id2chapter
    predictor.chapter2id = {v: k for k, v in id2chapter.items()}
    predictor.stage1_temperature = 1.0


def _make_s1_tokenizer(include_tti: bool):
    """Tokenizer that emits all three keys but only advertises tti if requested."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
        "token_type_ids": MagicMock(),
    }
    tok.model_input_names = (
        ["input_ids", "attention_mask", "token_type_ids"]
        if include_tti else
        ["input_ids", "attention_mask"]
    )
    return tok


def _inject_stage2_resolver(predictor, chapter: str, id2label: dict):
    """Add a mock Stage-2 resolver for the given chapter."""
    ch_tokenizer = MagicMock()
    ch_tokenizer.model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    ch_tokenizer.return_value = {k: MagicMock() for k in ch_tokenizer.model_input_names}
    predictor.stage2_models[chapter] = MagicMock()
    predictor.stage2_tokenizers[chapter] = ch_tokenizer
    predictor.stage2_id2label[chapter] = id2label
    predictor.stage2_temperatures[chapter] = 1.0


# ---------------------------------------------------------------------------
# Token filter tests
# ---------------------------------------------------------------------------

class TestTokenTypeIdsFilter:

    def test_bert_tokenizer_keeps_token_type_ids(self, mock_imports):
        """BERT tokenizer (tti in model_input_names) → tti passed to model."""
        tokenizer = _make_s1_tokenizer(include_tti=True)
        model = MagicMock()

        predictor = _make_predictor()
        _inject_stage1(predictor, tokenizer, model, {0: "A"})
        predictor.predict("Assessment: chest pain Plan: ECG", preprocessed=True)

        assert model.called, "Stage-1 model was not called"
        assert "token_type_ids" in model.call_args.kwargs

    def test_roberta_tokenizer_drops_token_type_ids(self, mock_imports):
        """RoBERTa tokenizer (no tti in model_input_names) → tti dropped."""
        tokenizer = _make_s1_tokenizer(include_tti=False)
        model = MagicMock()

        predictor = _make_predictor()
        _inject_stage1(predictor, tokenizer, model, {0: "A"})
        predictor.predict("Assessment: cough Plan: CXR", preprocessed=True)

        assert model.called, "Stage-1 model was not called"
        assert "token_type_ids" not in model.call_args.kwargs
        assert "input_ids" in model.call_args.kwargs
        assert "attention_mask" in model.call_args.kwargs


# ---------------------------------------------------------------------------
# use_reranker flag tests
# ---------------------------------------------------------------------------

class TestUseRerankerFlag:

    def test_use_reranker_false_skips_graph_load(self, mock_imports):
        """use_reranker=False → GraphReranker never instantiated."""
        _make_predictor(use_reranker=False)
        mock_imports.reranker_mod.GraphReranker.assert_not_called()

    def test_use_reranker_false_low_confidence_returns_resolver(self, mock_imports):
        """Low confidence + use_reranker=False → stage2_source='resolver'."""
        low_conf = np.array([[0.3, 0.2, 0.2, 0.2, 0.1]])
        mock_imports.torch.softmax.return_value.cpu.return_value.numpy.return_value = low_conf

        predictor = _make_predictor(use_reranker=False)
        _inject_stage1(predictor, MagicMock(), MagicMock(), {0: "A"})
        _inject_stage2_resolver(predictor, "A", {0: "A00", 1: "A01", 2: "A02", 3: "A03", 4: "A04"})

        result = predictor.predict("Assessment: pain Plan: rest", preprocessed=True)
        assert result["stage2_source"] == "resolver"


# ---------------------------------------------------------------------------
# Skip chapter tests
# ---------------------------------------------------------------------------

class TestSkipChapters:

    @pytest.mark.parametrize("skip_ch", ["U", "P", "Q"])
    def test_skip_chapter_returns_fallback(self, mock_imports, skip_ch):
        """Chapters U/P/Q → stage2_source='fallback' with registered default code."""
        predictor = _make_predictor()
        _inject_stage1(predictor, MagicMock(), MagicMock(), {0: skip_ch})
        predictor.skip_chapter_defaults = {skip_ch: f"{skip_ch}99.9"}

        result = predictor.predict("Assessment: Z Plan: routine", preprocessed=True)

        assert result["stage2_source"] == "fallback"
        assert result["chapter"] == skip_ch
        assert result["codes"] == [f"{skip_ch}99.9"]

    def test_skip_chapter_unknown_when_no_default(self, mock_imports):
        """Skip chapter with None default → codes=['UNKNOWN']."""
        predictor = _make_predictor()
        _inject_stage1(predictor, MagicMock(), MagicMock(), {0: "U"})
        predictor.skip_chapter_defaults = {"U": None}

        result = predictor.predict("note", preprocessed=True)

        assert result["stage2_source"] == "fallback"
        assert result["codes"] == ["UNKNOWN"]


# ---------------------------------------------------------------------------
# Missing Stage-2 model fallback
# ---------------------------------------------------------------------------

class TestMissingStage2Model:

    def test_no_resolver_returns_fallback_no_model(self, mock_imports):
        """Chapter with no loaded resolver → stage2_source='fallback_no_model'."""
        predictor = _make_predictor()
        _inject_stage1(predictor, MagicMock(), MagicMock(), {0: "X"})
        assert "X" not in predictor.stage2_models

        result = predictor.predict("Assessment: unknown Plan: refer", preprocessed=True)

        assert result["stage2_source"] == "fallback_no_model"
        assert result["codes"] == ["UNKNOWN"]
        assert result["chapter"] == "X"


# ---------------------------------------------------------------------------
# Module-level predict() cache
# ---------------------------------------------------------------------------

class TestPredictCache:

    def test_predict_cache_reuses_predictor(self, mock_imports):
        """Two predict() calls with same args → constructor called once."""
        sys.modules.pop("src.inference", None)
        from src import inference as inf_module
        inf_module._PREDICTOR_CACHE.clear()

        with patch.object(inf_module, "HierarchicalPredictor",
                          wraps=inf_module.HierarchicalPredictor) as mock_cls:
            for _ in range(2):
                try:
                    inf_module.predict("note", experiment_name="E-X", stage1_experiment="E-Y")
                except Exception:
                    pass
            assert mock_cls.call_count == 1

    def test_predict_cache_different_args_creates_new_predictor(self, mock_imports):
        """Different experiment args → separate constructor call per pair."""
        sys.modules.pop("src.inference", None)
        from src import inference as inf_module
        inf_module._PREDICTOR_CACHE.clear()

        with patch.object(inf_module, "HierarchicalPredictor",
                          wraps=inf_module.HierarchicalPredictor) as mock_cls:
            for exp in ["E-X", "E-Y"]:
                try:
                    inf_module.predict("note", experiment_name=exp, stage1_experiment="E-003")
                except Exception:
                    pass
            assert mock_cls.call_count == 2