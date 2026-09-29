"""FastAPI Router for Open-Vocabulary Object Detection using OWL-ViT."""

import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import (
    get_open_vocab_engine,
    get_owlvit_detector,
    validate_and_read_image_upload,
)
from src.api.errors import ImageValidationError
from src.api.schemas import (
    OpenVocabDetectionResponse,
    OpenVocabUnloadResponse,
    RegionRecognitionResponse,
)
from src.phase6.config import (
    DEFAULT_CROP_PADDING_PERCENT,
    DEFAULT_MAX_REGIONS,
    MAX_ALLOWED_REGIONS,
)
from src.phase6.engine import OpenVocabEngine
from src.phase6.owlvit.config import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
)
from src.phase6.owlvit.detector import OWLViTDetector
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager
from src.phase6.region_pipeline import RegionRecognitionPipeline

logger = logging.getLogger("api.open_vocab_detection")
router = APIRouter(tags=["Open-Vocabulary Object Detection & Region Recognition"])

# Bounded concurrency lock to prevent CPU saturation on host
_inference_semaphore = asyncio.Semaphore(1)


@router.post(
    "/open-vocabulary/detect",
    response_model=OpenVocabDetectionResponse,
    summary="Open-Vocabulary Object Detection & Bounding Boxes",
    description="Detects and localizes arbitrary visual objects in images using Google OWL-ViT with natural language queries.",
)
async def detect_open_vocabulary_objects(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    text_queries: str = Form(
        ...,
        description="Comma-separated or JSON list of candidate object queries (e.g. 'dog, person, red car')",
    ),
    top_k: int = Form(
        DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Maximum number of detection bounding boxes to return",
    ),
    score_threshold: float = Form(
        DEFAULT_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum detection alignment score cutoff [0.0, 1.0]",
    ),
    prompt_template: str = Form(
        DEFAULT_PROMPT_TEMPLATE,
        description="Prompt template string containing '{}' (e.g., 'a photo of a {}')",
    ),
    detector: OWLViTDetector = Depends(get_owlvit_detector),
) -> OpenVocabDetectionResponse:
    """Execute open-vocabulary object detection and bounding box prediction with OWL-ViT."""
    req_id = getattr(request.state, "request_id", None)

    # 1. Parse text_queries input
    queries: list[str] = []
    raw_text = text_queries.strip()
    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            queries = [q.strip() for q in raw_text.split(",") if q.strip()]
    else:
        queries = [q.strip() for q in raw_text.split(",") if q.strip()]

    if not queries:
        raise ImageValidationError(
            "At least one non-empty text query must be provided.",
            code="INVALID_TEXT_QUERIES",
            request_id=req_id,
        )

    if len(queries) > MAX_TEXT_QUERIES:
        raise ImageValidationError(
            f"Maximum of {MAX_TEXT_QUERIES} text queries allowed (received {len(queries)}).",
            code="TOO_MANY_QUERIES",
            request_id=req_id,
        )

    for q in queries:
        if len(q) > MAX_TEXT_QUERY_LENGTH:
            raise ImageValidationError(
                f"Query '{q[:20]}...' exceeds maximum length of {MAX_TEXT_QUERY_LENGTH} characters.",
                code="QUERY_TOO_LONG",
                request_id=req_id,
            )

    # 2. Validate and stream-read image upload
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 3. Execute inference under concurrency semaphore
    async with _inference_semaphore:
        result = await run_in_threadpool(
            detector.detect_objects,
            image_input=pil_img,
            text_queries=queries,
            score_threshold=score_threshold,
            top_k=top_k,
            prompt_template=prompt_template,
        )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "OWL-ViT object detection failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    # Attach client filename and bytes
    if "image" in result:
        result["image"]["file_name"] = filename
        result["image"]["file_size_bytes"] = len(raw_bytes)

    return OpenVocabDetectionResponse.model_validate(result)


@router.post(
    "/open-vocabulary/region-recognition",
    response_model=RegionRecognitionResponse,
    summary="Two-Stage Grounded Region-Level Open-Vocabulary Recognition",
    description="Detects visual entities with OWL-ViT, extracts 10% context-padded region crops, and executes batched OpenCLIP semantic ranking across all regions.",
)
async def recognize_open_vocabulary_regions(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    text_queries: str = Form(
        ...,
        description="Comma-separated or JSON list of candidate object queries (e.g. 'dog, person, red car')",
    ),
    top_k: int = Form(
        DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="Maximum number of refined OpenCLIP query rankings to return per region",
    ),
    detection_threshold: float = Form(
        DEFAULT_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum detection alignment score cutoff for OWL-ViT region grounding [0.0, 1.0]",
    ),
    similarity_threshold: Optional[float] = Form(
        None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff for OpenCLIP region matches",
    ),
    max_regions: int = Form(
        DEFAULT_MAX_REGIONS,
        ge=1,
        le=MAX_ALLOWED_REGIONS,
        description="Maximum candidate regions to extract and analyze",
    ),
    crop_padding_percent: int = Form(
        DEFAULT_CROP_PADDING_PERCENT,
        ge=0,
        le=50,
        description="Symmetrical context padding percentage added to each bounding box (default 10%)",
    ),
    prompt_template: str = Form(
        DEFAULT_PROMPT_TEMPLATE,
        description="Prompt template string containing '{}' (e.g., 'a photo of a {}')",
    ),
    detector: OWLViTDetector = Depends(get_owlvit_detector),
    open_vocab_engine: OpenVocabEngine = Depends(get_open_vocab_engine),
) -> RegionRecognitionResponse:
    """Execute two-stage grounded region-level open-vocabulary recognition."""
    req_id = getattr(request.state, "request_id", None)

    # 1. Parse text_queries input
    queries: list[str] = []
    raw_text = text_queries.strip()
    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            queries = [q.strip() for q in raw_text.split(",") if q.strip()]
    else:
        queries = [q.strip() for q in raw_text.split(",") if q.strip()]

    if not queries:
        raise ImageValidationError(
            "At least one non-empty text query must be provided.",
            code="INVALID_TEXT_QUERIES",
            request_id=req_id,
        )

    if len(queries) > MAX_TEXT_QUERIES:
        raise ImageValidationError(
            f"Maximum of {MAX_TEXT_QUERIES} text queries allowed (received {len(queries)}).",
            code="TOO_MANY_QUERIES",
            request_id=req_id,
        )

    for q in queries:
        if len(q) > MAX_TEXT_QUERY_LENGTH:
            raise ImageValidationError(
                f"Query '{q[:20]}...' exceeds maximum length of {MAX_TEXT_QUERY_LENGTH} characters.",
                code="QUERY_TOO_LONG",
                request_id=req_id,
            )

    # 2. Validate and stream-read image upload
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 3. Instantiate Region Pipeline using existing singleton instances
    pipeline = RegionRecognitionPipeline(
        detector=detector,
        classifier=open_vocab_engine._classifier,
    )

    # 4. Execute inference under concurrency semaphore
    async with _inference_semaphore:
        result = await run_in_threadpool(
            pipeline.process_image,
            image_input=pil_img,
            text_queries=queries,
            detection_threshold=detection_threshold,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            max_regions=max_regions,
            crop_padding_percent=crop_padding_percent,
            prompt_template=prompt_template,
        )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "Region-level recognition failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    # Attach client filename and bytes
    if "image" in result:
        result["image"]["file_name"] = filename
        result["image"]["file_size_bytes"] = len(raw_bytes)

    return RegionRecognitionResponse.model_validate(result)



@router.post(
    "/open-vocabulary/unload",
    response_model=OpenVocabUnloadResponse,
    summary="Unload OWL-ViT Model",
    description="Explicitly releases OWL-ViT from RAM and runs garbage collection to reclaim host memory.",
)
async def unload_owl_vit_model() -> OpenVocabUnloadResponse:
    """Manually release OWL-ViT model from RAM."""
    was_unloaded = OWLViTLifecycleManager.unload()
    msg = (
        "OWL-ViT model was successfully unloaded from memory."
        if was_unloaded
        else "OWL-ViT model was already unloaded."
    )
    return OpenVocabUnloadResponse(
        success=True,
        message=msg,
        unloaded=was_unloaded,
    )


@router.get(
    "/open-vocabulary/status",
    summary="OWL-ViT Runtime Lifecycle Status",
    description="Returns telemetry regarding whether OWL-ViT is loaded, parameter counts, and inference history.",
)
async def get_owl_vit_status() -> dict:
    """Retrieve runtime memory and lifecycle status."""
    return OWLViTLifecycleManager.get_status()
