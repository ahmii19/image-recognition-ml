"""Health and Readiness API Routes."""

from fastapi import APIRouter, Depends, Request
from src.api.config import API_VERSION
from src.api.dependencies import get_recognition_engine
from src.api.schemas import HealthResponse, ReadyResponse
from src.phase3.engine import AdvancedRecognitionEngine

router = APIRouter(tags=["Health & Telemetry"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Liveness Check",
    description="Returns immediate service status and health indicators.",
)
async def health_check(request: Request) -> HealthResponse:
    """Liveness probe to verify FastAPI process status."""
    engine: AdvancedRecognitionEngine = getattr(request.app.state, "engine", None)
    classifier_active = engine is not None and engine._classifier is not None
    detector_active = engine is not None and engine._detector is not None

    return HealthResponse(
        status="ok",
        service="image-recognition-api",
        version=API_VERSION,
        engine="ready" if (classifier_active and detector_active) else "initializing",
        models={
            "classifier": classifier_active,
            "detector": detector_active,
        },
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Model Readiness Probe",
    description="Validates that ML backbones (MobileNetV2 and SSD-MobileNetV2) are active and ready for inference.",
)
async def readiness_check(
    engine: AdvancedRecognitionEngine = Depends(get_recognition_engine),
) -> ReadyResponse:
    """Readiness probe to confirm ML models are loaded and receptive to inference traffic."""
    classifier_loaded = engine._classifier is not None
    detector_loaded = engine._detector is not None
    ready = classifier_loaded and detector_loaded

    return ReadyResponse(
        status="ready" if ready else "not_ready",
        engine="AdvancedRecognitionEngine-v3",
        models_ready=ready,
        classifier_loaded=classifier_loaded,
        detector_loaded=detector_loaded,
    )
