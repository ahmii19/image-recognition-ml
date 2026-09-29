"""Recognition, Classification, and Detection Endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import get_recognition_engine, validate_and_read_image_upload
from src.api.errors import (
    ImageValidationError,
    InvalidModeError,
)
from src.api.schemas import (
    ClassificationResult,
    DetectionResult,
    RecognitionResponse,
)
from src.phase3.config import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
)
from src.phase3.engine import AdvancedRecognitionEngine

logger = logging.getLogger("api.recognition")
router = APIRouter(tags=["Image Recognition"])


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Unified Image Recognition",
    description="Processes uploaded image through classification, object detection, or both simultaneously.",
)
async def recognize_image(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    mode: str = Form(
        "all",
        description="Recognition mode: 'all', 'classification', or 'detection'",
    ),
    classification_threshold: float = Form(
        DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for classification",
    ),
    detection_threshold: float = Form(
        DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence cutoff for object bounding boxes",
    ),
    render_visualization: bool = Form(
        False,
        description="Whether to generate and save bounding box visualization image",
    ),
    top_k: int = Form(
        5,
        ge=1,
        le=20,
        description="Number of top classification candidates to return",
    ),
    engine: AdvancedRecognitionEngine = Depends(get_recognition_engine),
) -> RecognitionResponse:
    """Execute unified image recognition using Phase 3 ML Engine."""
    req_id = getattr(request.state, "request_id", None)
    mode_normalized = mode.strip().lower()

    if mode_normalized not in ("all", "classification", "detection"):
        raise InvalidModeError(
            f"Invalid mode '{mode}'. Choose from 'all', 'classification', or 'detection'.",
            request_id=req_id,
        )

    # 1. Validate and load image
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 2. Execute CPU-bound inference in threadpool
    result = await run_in_threadpool(
        engine.recognize,
        image_input=pil_img,
        mode=mode_normalized,
        classification_threshold=classification_threshold,
        detection_threshold=detection_threshold,
        render_visualization=render_visualization,
        top_k=top_k,
    )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "Image recognition failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    # Attach the original client filename for clarity
    if "image" in result:
        result["image"]["file_name"] = filename
        result["image"]["file_size_bytes"] = len(raw_bytes)

    return RecognitionResponse.model_validate(result)


@router.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Image Classification",
    description="Performs multi-class ImageNet classification returning Top-K predicted labels with confidence scores.",
)
async def classify_image(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    top_k: int = Form(
        5,
        ge=1,
        le=20,
        description="Number of top classification candidates to return",
    ),
    threshold: float = Form(
        DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Classification certainty threshold",
    ),
    engine: AdvancedRecognitionEngine = Depends(get_recognition_engine),
) -> ClassificationResult:
    """Execute standalone ImageNet classification using MobileNetV2."""
    req_id = getattr(request.state, "request_id", None)

    # 1. Validate and load image
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 2. Execute CPU-bound classification in threadpool
    result = await run_in_threadpool(
        engine.classifier.classify,
        image_input=pil_img,
        top_k=top_k,
        threshold=threshold,
    )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "Classification failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    return ClassificationResult.model_validate(result)


@router.post(
    "/detect",
    response_model=DetectionResult,
    summary="Object Detection",
    description="Performs COCO-80 object detection returning detected entities, bounding boxes, and confidence levels.",
)
async def detect_image(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    threshold: float = Form(
        DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Object detection confidence cutoff",
    ),
    max_detections: int = Form(
        20,
        ge=1,
        le=100,
        description="Maximum number of detection bounding boxes to return",
    ),
    engine: AdvancedRecognitionEngine = Depends(get_recognition_engine),
) -> DetectionResult:
    """Execute standalone object detection using SSD-MobileNetV2."""
    req_id = getattr(request.state, "request_id", None)

    # 1. Validate and load image
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 2. Execute CPU-bound detection in threadpool
    result = await run_in_threadpool(
        engine.detector.detect,
        image_input=pil_img,
        threshold=threshold,
        max_detections=max_detections,
    )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "Object detection failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    return DetectionResult.model_validate(result)
