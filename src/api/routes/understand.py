"""FastAPI Router for Phase 7 General Image Understanding (VLM — Qwen2.5-VL-3B-Instruct).

This module is a THIN API ADAPTER over the Phase 7B VLMEngine/VLMLifecycleManager.
No model loading, inference implementation, or parsing logic lives here.

Architecture:
    Client → POST /api/v1/understand
           → validate_and_read_image_upload() (existing dep — reused)
           → asyncio.Semaphore(1) (bounded concurrency)
           → run_in_threadpool() (non-blocking async)
           → VLMLifecycleManager.get_engine() (lazy singleton)
           → VLMEngine.understand(pil_image, mode)
           → Pydantic validation → JSON response

Endpoints:
    POST   /api/v1/understand         — General image understanding
    POST   /api/v1/understand/unload  — Explicit VLM memory release
    GET    /api/v1/understand/status  — VLM lifecycle telemetry
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import validate_and_read_image_upload
from src.api.errors import (
    APIException,
    ImageValidationError,
    ModelUnavailableError,
)
from src.api.schemas import (
    VLMParseMeta,
    VLMTimingInfo,
    VLMImageInfo,
    VLMUnderstandingResponse,
    VLMStatusResponse,
    VLMUnloadResponse,
)
from src.phase7.config import VLM_MODEL_ID
from src.phase7.lifecycle import VLMLifecycleManager
from src.phase7.prompts import PromptMode, list_prompt_modes

logger = logging.getLogger("api.understand")
router = APIRouter(tags=["General Image Understanding (Phase 7 VLM)"])

# Bounded concurrency: VLM inference is very memory-intensive.
# Semaphore(1) mirrors the pattern used in open_vocab_detection.py.
# This prevents concurrent VLM inference from exhausting VRAM or RAM.
_vlm_semaphore = asyncio.Semaphore(1)

# Valid prompt mode values (resolved once at import time)
_VALID_MODES = {m.value for m in PromptMode if m != PromptMode.CUSTOM}


def _get_vlm_engine():
    """Retrieve the VLMEngine singleton via VLMLifecycleManager (lazy-loaded).

    Never loads the model at application startup — only on first inference request.

    Raises:
        ModelUnavailableError (503): If the VLM cannot be initialized.
    """
    try:
        return VLMLifecycleManager.get_engine()
    except RuntimeError as exc:
        # CUDA explicitly requested but unavailable (from src.device.resolve_device)
        raise ModelUnavailableError(
            f"VLM device initialization failed: {exc}"
        )
    except Exception as exc:
        raise ModelUnavailableError(
            f"VLM General Image Understanding engine could not be initialized: {exc}"
        )


@router.post(
    "/understand",
    response_model=VLMUnderstandingResponse,
    summary="General Image Understanding (VLM)",
    description=(
        "Accepts any image and returns structured semantic understanding using the "
        "Qwen2.5-VL-3B-Instruct Vision-Language Model. Supports photographs, illustrations, "
        "documents, diagrams, charts, maps, medical images, and mixed-content images. "
        "The VLM is lazy-loaded on first request."
    ),
    responses={
        400: {"description": "Invalid image (corrupted, empty, unsupported format)"},
        413: {"description": "Image upload exceeds size limit"},
        422: {"description": "Invalid prompt mode or request parameters"},
        500: {"description": "Unexpected inference or parsing failure"},
        503: {"description": "VLM model unavailable or cannot be loaded"},
    },
)
async def general_image_understanding(
    request: Request,
    image: UploadFile = File(
        ...,
        description="Multipart image file (JPG, PNG, WEBP, BMP). Validated before VLM inference.",
    ),
    mode: str = Form(
        "general",
        description=(
            "Prompt mode controlling the analysis type. "
            "Options: general, detailed, document, diagram, chart, map, medical, brief. "
            "Default: general."
        ),
    ),
    max_tokens: int = Form(
        512,
        ge=32,
        le=2048,
        description="Maximum number of new tokens the VLM may generate (32–2048). Default: 512.",
    ),
) -> VLMUnderstandingResponse:
    """Run general image understanding via the Phase 7B VLM engine.

    The endpoint:
    1. Validates the uploaded image (size, format, decodability).
    2. Validates the prompt mode.
    3. Acquires the concurrency semaphore.
    4. Runs VLM inference in a threadpool (non-blocking).
    5. Returns Pydantic-validated structured JSON.
    """
    req_id = getattr(request.state, "request_id", None)

    # ── 1. Validate prompt mode ───────────────────────────────────────────────
    mode_stripped = mode.strip().lower()
    if mode_stripped not in _VALID_MODES:
        raise APIException(
            status_code=422,
            code="INVALID_VLM_MODE",
            message=(
                f"Invalid prompt mode '{mode}'. "
                f"Supported modes: {', '.join(sorted(_VALID_MODES))}."
            ),
            request_id=req_id,
        )
    prompt_mode = PromptMode(mode_stripped)

    # ── 2. Validate and stream-read image upload (reuse existing dep) ─────────
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(
        image, request=request
    )

    # ── 3. Lazy-load VLM engine singleton (never loads at startup) ────────────
    try:
        engine = _get_vlm_engine()
    except ModelUnavailableError:
        raise
    except Exception as exc:
        logger.error("[%s] Unexpected error during VLM engine retrieval: %s", req_id, exc, exc_info=True)
        raise ModelUnavailableError(
            "VLM engine is currently unavailable. Please try again later."
        )

    # ── 4. Execute VLM inference (non-blocking via threadpool) ───────────────
    logger.info(
        "[%s] VLM understand request: file=%s size=%d mode=%s max_tokens=%d device=%s",
        req_id, filename, len(raw_bytes), mode_stripped, max_tokens, engine.device,
    )

    async with _vlm_semaphore:
        try:
            result = await run_in_threadpool(
                engine.understand,
                image_input=pil_img,
                mode=prompt_mode,
                max_new_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error(
                "[%s] Unexpected error during VLM inference: %s", req_id, exc, exc_info=True
            )
            raise APIException(
                status_code=500,
                code="VLM_INFERENCE_ERROR",
                message="An unexpected error occurred during VLM inference. Please try again.",
                request_id=req_id,
            )

    # ── 5. Handle VLM-level failures ─────────────────────────────────────────
    if not result.get("success", False):
        error_msg = result.get("error") or "VLM inference failed without a specific error."
        logger.warning("[%s] VLM inference returned success=False: %s", req_id, error_msg)

        # Classify engine-reported errors into correct HTTP status codes
        error_lower = error_msg.lower()
        if "not loaded" in error_lower or "unavailable" in error_lower:
            raise ModelUnavailableError(error_msg, request_id=req_id)
        if "image loading" in error_lower or "corrupted" in error_lower:
            raise ImageValidationError(error_msg, code="IMAGE_PROCESSING_ERROR", request_id=req_id)

        raise APIException(
            status_code=500,
            code="VLM_INFERENCE_FAILED",
            message=error_msg,
            request_id=req_id,
        )

    # ── 6. Build validated Pydantic response ──────────────────────────────────
    timing_raw = result.get("timing", {})
    parse_meta_raw = result.get("parse_meta", {})
    image_info_raw = result.get("image_info", {})

    logger.info(
        "[%s] VLM inference complete: mode=%s total_ms=%.1f recovered=%s",
        req_id,
        result.get("mode"),
        timing_raw.get("total_ms", 0.0),
        parse_meta_raw.get("recovered", False),
    )

    return VLMUnderstandingResponse(
        success=True,
        mode=result["mode"],
        model_id=result["model_id"],
        device=result["device"],
        image_info=VLMImageInfo(
            width=image_info_raw.get("width", pil_img.width),
            height=image_info_raw.get("height", pil_img.height),
            mode=image_info_raw.get("mode", pil_img.mode),
        ),
        understanding=result.get("understanding", {}),
        parse_meta=VLMParseMeta(
            recovered=parse_meta_raw.get("recovered", False),
            recovery_method=parse_meta_raw.get("recovery_method", "direct"),
            missing_keys=parse_meta_raw.get("missing_keys", []),
            extra_keys=parse_meta_raw.get("extra_keys", []),
            warnings=parse_meta_raw.get("warnings", []),
        ),
        timing=VLMTimingInfo(
            preprocessing_ms=timing_raw.get("preprocessing_ms", 0.0),
            inference_ms=timing_raw.get("inference_ms", 0.0),
            decoding_ms=timing_raw.get("decoding_ms", 0.0),
            parsing_ms=timing_raw.get("parsing_ms", 0.0),
            total_ms=timing_raw.get("total_ms", 0.0),
        ),
        request_id=req_id,
    )


@router.post(
    "/understand/unload",
    response_model=VLMUnloadResponse,
    summary="Unload VLM from Memory",
    description=(
        "Explicitly releases the Qwen2.5-VL-3B-Instruct model from GPU/CPU memory "
        "and runs garbage collection. Useful after a batch inference session on Colab T4."
    ),
)
async def unload_vlm_model() -> VLMUnloadResponse:
    """Explicitly release the VLM model singleton from memory."""
    was_unloaded = VLMLifecycleManager.unload()
    msg = (
        "VLM model was successfully unloaded from memory."
        if was_unloaded
        else "VLM model was already unloaded (no action taken)."
    )
    logger.info("VLM explicit unload requested: was_unloaded=%s", was_unloaded)
    return VLMUnloadResponse(success=True, message=msg, unloaded=was_unloaded)


@router.get(
    "/understand/status",
    response_model=VLMStatusResponse,
    summary="VLM Runtime Lifecycle Status",
    description=(
        "Returns telemetry regarding whether the Phase 7 VLM is loaded, "
        "parameter counts, inference history, device, and lifecycle configuration."
    ),
)
async def get_vlm_status() -> VLMStatusResponse:
    """Retrieve VLM runtime lifecycle and telemetry status."""
    status = VLMLifecycleManager.get_status()
    return VLMStatusResponse(**status)


@router.get(
    "/understand/modes",
    summary="List Available VLM Prompt Modes",
    description="Returns all supported prompt modes for POST /api/v1/understand.",
)
async def list_understand_modes() -> dict:
    """Return all available VLM prompt modes and their descriptions."""
    return {
        "available_modes": list_prompt_modes(),
        "default_mode": "general",
    }
