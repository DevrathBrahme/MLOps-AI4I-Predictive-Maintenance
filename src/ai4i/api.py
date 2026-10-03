"""FastAPI service: advisory failure-mode recommendations for a human operator.

Scores one sensor reading, logs the recommendation to predictions_log, and only
then returns it. The service never acts on a prediction; an operator decides.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from ai4i.db import get_connection
from ai4i.features import build_features
from ai4i.model import CLASSES
from ai4i.prediction_log import PROBA_COLUMNS, log_prediction
from ai4i.registry import REGISTERED_MODEL_NAME, resolve_champion
from ai4i.risk import THRESHOLD_TAG, assign_tiers, confidence_score
from ai4i.schemas import (
    ClassProbabilities,
    PhysicsFeatures,
    Recommendation,
    SensorReading,
)

PHYSICS_COLUMNS = tuple(PhysicsFeatures.model_fields)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServedModel:
    """The champion exactly as loaded at startup."""

    name: str
    version: int
    threshold: float
    pipeline: Pipeline


def load_champion() -> ServedModel:
    """Resolve the champion alias once and load exactly that version.

    Raises if the alias or the threshold tag is missing, so the service never
    starts without a registered model and its pre-registered operating point.
    """
    version = resolve_champion(MlflowClient())
    if THRESHOLD_TAG not in version.tags:
        raise RuntimeError(
            f"{REGISTERED_MODEL_NAME} version {version.version} has no "
            f"{THRESHOLD_TAG} tag; run python -m ai4i.thresholds first"
        )
    pipeline = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{version.version}"
    )
    return ServedModel(
        name=REGISTERED_MODEL_NAME,
        version=int(version.version),
        threshold=float(version.tags[THRESHOLD_TAG]),
        pipeline=pipeline,
    )


def top_two(row: np.ndarray) -> tuple[int, int]:
    """Return (predicted, runner-up) class codes; runner-up by masked argmax (M5)."""
    predicted = int(np.argmax(row))
    masked = row.copy()
    masked[predicted] = -np.inf
    return predicted, int(np.argmax(masked))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion once at startup; the app refuses to start without it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.state.served = load_champion()
    logger.info(
        "Serving %s version %s (elevated threshold %r)",
        app.state.served.name, app.state.served.version, app.state.served.threshold,
    )
    yield


app = FastAPI(
    title="AI4I failure-mode recommendations",
    description=(
        "Scores a machine reading and returns a flagged recommendation with a "
        "confidence score for a human operator. Advisory only: the service never "
        "triggers an action; an operator decides."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    """Report the served model and whether predictions can be logged."""
    served: ServedModel = request.app.state.served
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unreachable: recommendations cannot be logged",
        ) from exc
    return {
        "status": "ok",
        "model_name": served.name,
        "model_version": served.version,
        "elevated_threshold": served.threshold,
    }


@app.post("/predict")
def predict(reading: SensorReading, request: Request) -> Recommendation:
    """Score one reading, log the recommendation, then return it.

    Returns 503 if the recommendation cannot be logged: no operator ever sees
    a recommendation that is missing from predictions_log.
    """
    served: ServedModel = request.app.state.served
    raw = reading.model_dump()
    features = build_features(pd.DataFrame([raw]))
    proba = served.pipeline.predict_proba(features)
    row = proba[0]
    predicted, runner_up = top_two(row)
    tier = str(assign_tiers(proba, served.threshold)[0])
    score = float(confidence_score(proba)[0])
    physics = {name: float(features.at[0, name]) for name in PHYSICS_COLUMNS}

    record = {
        "model_name": served.name,
        "model_version": served.version,
        **raw,
        **physics,
        **{column: float(p) for column, p in zip(PROBA_COLUMNS, row)},
        "predicted_mode": CLASSES[predicted],
        "runner_up_mode": CLASSES[runner_up],
        "confidence_score": score,
        "elevated_threshold": served.threshold,
        "risk_tier": tier,
    }
    try:
        with get_connection() as conn:
            request_id, created_at = log_prediction(conn, record)
    except psycopg.OperationalError as exc:
        logger.error("Recommendation not logged, returning 503: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Recommendation could not be logged, so none is returned",
        ) from exc

    return Recommendation(
        request_id=request_id,
        created_at=created_at,
        model_name=served.name,
        model_version=served.version,
        flagged=tier != "low",
        risk_tier=tier,
        confidence_score=score,
        predicted_mode=CLASSES[predicted],
        predicted_probability=float(row[predicted]),
        runner_up_mode=CLASSES[runner_up],
        runner_up_probability=float(row[runner_up]),
        class_probabilities=ClassProbabilities(**dict(zip(CLASSES, map(float, row)))),
        physics_features=PhysicsFeatures(**physics),
    )