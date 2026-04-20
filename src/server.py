"""
server.py — ICD-10 Prediction Model Server
===========================================
FastAPI application that loads the calibrated hierarchical pipeline once at
startup and serves predictions over HTTP. Designed for local development and
demo use — add auth middleware before any public deployment.

Endpoints
---------
    GET  /health          Liveness + model readiness check
    GET  /info            Experiment names, threshold, chapter count
    POST /predict         Single-note ICD-10 prediction
    POST /predict/batch   Batch prediction (list of notes)

Usage (via scripts/serve.py)
-----------------------------
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8080 --threshold 0.6
    uv run python scripts/serve.py --reload   # auto-reload on code changes
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Deferred import — keeps module importable even before HierarchicalPredictor
# is instantiated, so FastAPI can start and report errors cleanly.
# ---------------------------------------------------------------------------
_predictor = None
_server_config: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    note: str = Field(
        ...,
        min_length=10,
        description="Clinical note text. SOAP format preferred; APSO-Flip and "
                    "ICD-10 redaction are applied automatically.",
        examples=["S: Patient presents with left knee pain worsening over 3 months. "
                  "O: Swelling on exam, limited ROM. A: Likely OA. P: NSAIDs, ortho referral."],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top ICD-10 predictions to return.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for auto-code decision. "
                    "Defaults to the server's configured threshold.",
    )


class PredictionItem(BaseModel):
    rank:       int
    code:       str
    confidence: float


class PredictResponse(BaseModel):
    top_code:      str
    confidence:    float
    chapter:       str
    chapter_name:  str
    decision:      str   # "AUTO_CODE" or "HUMAN_REVIEW"
    threshold:     float
    predictions:   list[PredictionItem]
    stage2_source: str
    latency_ms:    float


class BatchPredictRequest(BaseModel):
    notes: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of clinical notes to predict (max 100 per request).",
    )
    top_k: int = Field(default=3, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    experiment:      str
    stage1_experiment: str
    chapters_loaded: int
    device:          str


class InfoResponse(BaseModel):
    experiment:        str
    stage1_experiment: str
    threshold:         float
    chapters_loaded:   int
    device:            str
    top_k_default:     int


# ---------------------------------------------------------------------------
# Chapter descriptions
# ---------------------------------------------------------------------------

CHAPTER_NAMES: dict[str, str] = {
    "A": "Certain infectious and parasitic diseases",
    "B": "Certain infectious and parasitic diseases",
    "C": "Neoplasms",
    "D": "Diseases of blood and blood-forming organs",
    "E": "Endocrine, nutritional and metabolic diseases",
    "F": "Mental and behavioural disorders",
    "G": "Diseases of the nervous system",
    "H": "Diseases of the eye / ear",
    "I": "Diseases of the circulatory system",
    "J": "Diseases of the respiratory system",
    "K": "Diseases of the digestive system",
    "L": "Diseases of the skin and subcutaneous tissue",
    "M": "Diseases of the musculoskeletal system",
    "N": "Diseases of the genitourinary system",
    "O": "Pregnancy, childbirth and the puerperium",
    "P": "Certain conditions originating in the perinatal period",
    "Q": "Congenital malformations and chromosomal abnormalities",
    "R": "Symptoms, signs and abnormal clinical findings",
    "S": "Injury, poisoning — external causes",
    "T": "Poisoning by drugs and toxic substances",
    "U": "Codes for special purposes",
    "Z": "Factors influencing health status",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(confidence: float, threshold: float) -> str:
    return "AUTO_CODE" if confidence >= threshold else "HUMAN_REVIEW"


def _run_prediction(note: str, top_k: int, threshold: float) -> PredictResponse:
    """Run inference and return a structured response. Raises on empty note."""
    note = note.strip()
    if not note:
        raise HTTPException(status_code=422, detail="Note must not be empty.")

    t0 = time.perf_counter()
    result = _predictor.predict(note, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    codes         = result["codes"]
    scores        = result["scores"]
    chapter       = result.get("chapter", "?")
    stage2_source = result.get("stage2_source", "resolver")
    confidence    = float(scores[0]) if scores else 0.0

    return PredictResponse(
        top_code      = codes[0] if codes else "UNKNOWN",
        confidence    = round(confidence, 6),
        chapter       = chapter,
        chapter_name  = CHAPTER_NAMES.get(chapter, "Unknown chapter"),
        decision      = _make_decision(confidence, threshold),
        threshold     = threshold,
        predictions   = [
            PredictionItem(rank=i + 1, code=c, confidence=round(s, 6))
            for i, (c, s) in enumerate(zip(codes, scores))
        ],
        stage2_source = stage2_source,
        latency_ms    = round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Lifespan — models loaded once at startup, cleaned up on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the HierarchicalPredictor at startup; release on shutdown."""
    global _predictor

    from src.inference import HierarchicalPredictor

    experiment       = _server_config.get("experiment",        "E-004a_Hierarchical_E002Init")
    stage1_experiment = _server_config.get("stage1_experiment", "E-003_Hierarchical_ICD10")

    print(f"\n{'='*60}")
    print(f"  ICD-10 Model Server — loading pipeline...")
    print(f"  Experiment:  {experiment}")
    print(f"  Stage-1:     {stage1_experiment}")
    print(f"{'='*60}")

    _predictor = HierarchicalPredictor(
        experiment_name=experiment,
        stage1_experiment=stage1_experiment,
    )

    chapters = len(_predictor.stage2_models)
    print(f"\n✅ Pipeline ready — {chapters} resolvers loaded\n")

    yield  # server is running

    # Shutdown: release model memory
    _predictor = None
    print("\n🛑 Model server shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(
    experiment:        str   = "E-004a_Hierarchical_E002Init",
    stage1_experiment: str   = "E-003_Hierarchical_ICD10",
    default_threshold: float = 0.7,
    default_top_k:     int   = 5,
) -> FastAPI:
    """
    Factory function — creates and configures the FastAPI app.

    Called by scripts/serve.py with CLI arguments; also importable directly
    for testing or embedding in a larger application.
    """
    _server_config["experiment"]        = experiment
    _server_config["stage1_experiment"] = stage1_experiment
    _server_config["threshold"]         = default_threshold
    _server_config["top_k"]             = default_top_k

    app = FastAPI(
        title       = "ICD-10 Prediction API",
        description = (
            "Calibrated two-stage hierarchical ICD-10 coding pipeline. "
            "Accepts clinical notes in SOAP format and returns ranked ICD-10 "
            "predictions with a Use Case B routing decision."
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    # CORS — allow all origins for local dev; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Liveness and readiness check. Returns 503 if models not loaded."""
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")
        return HealthResponse(
            status           = "ok",
            model_loaded     = True,
            experiment       = _server_config["experiment"],
            stage1_experiment= _server_config["stage1_experiment"],
            chapters_loaded  = len(_predictor.stage2_models),
            device           = str(_predictor.device),
        )

    @app.get("/info", response_model=InfoResponse, tags=["System"])
    async def info():
        """Returns server configuration: experiment names, threshold, device."""
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")
        return InfoResponse(
            experiment        = _server_config["experiment"],
            stage1_experiment = _server_config["stage1_experiment"],
            threshold         = _server_config["threshold"],
            chapters_loaded   = len(_predictor.stage2_models),
            device            = str(_predictor.device),
            top_k_default     = _server_config["top_k"],
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
    async def predict(req: PredictRequest):
        """
        Predict ICD-10 codes for a single clinical note.

        The pipeline applies APSO-Flip and ICD-10 redaction automatically —
        pass the raw note as written, do not preprocess it first.

        Returns the top-k predictions ranked by confidence, plus a routing
        decision (AUTO_CODE / HUMAN_REVIEW) based on the configured threshold.
        """
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")

        threshold = req.threshold if req.threshold is not None else _server_config["threshold"]
        return _run_prediction(req.note, req.top_k, threshold)

    @app.post("/predict/batch", tags=["Prediction"])
    async def predict_batch(req: BatchPredictRequest):
        """
        Predict ICD-10 codes for a list of clinical notes (max 100).

        Returns a list of prediction objects in the same order as the input.
        Each item has the same structure as the single /predict response,
        plus an `index` field for correlation with the input list.
        """
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")

        threshold = req.threshold if req.threshold is not None else _server_config["threshold"]
        results = []
        for i, note in enumerate(req.notes):
            try:
                pred = _run_prediction(note, req.top_k, threshold)
                results.append({"index": i, **pred.model_dump()})
            except HTTPException as e:
                results.append({"index": i, "error": e.detail})
        return JSONResponse(content={"results": results, "count": len(results)})

    return app