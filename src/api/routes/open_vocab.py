"""Open-Vocabulary Recognition Endpoint using OpenCLIP ViT-B/32."""

import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import get_open_vocab_engine, validate_and_read_image_upload
from src.api.errors import ImageValidationError
from src.api.schemas import (
    OpenVocabResponse,
)
from src.phase6.config import (
    DEFAULT_PROMPT_TEMPLATE,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
)
from src.phase6.engine import OpenVocabEngine

logger = logging.getLogger("api.open_vocab")
router = APIRouter(tags=["Open-Vocabulary Recognition"])


@router.post(
    "/open-vocabulary",
    response_model=OpenVocabResponse,
    summary="Zero-Shot Open-Vocabulary Recognition",
    description="Performs zero-shot image-text recognition using OpenCLIP ViT-B/32 on arbitrary custom text concepts.",
)
async def open_vocabulary_recognition(
    request: Request,
    image: UploadFile = File(..., description="Multipart image file (JPG, PNG, WEBP, BMP)"),
    text_queries: str = Form(
        ...,
        description="Comma-separated or JSON list of candidate labels/concepts (e.g. 'rose, sunflower, cat')",
    ),
    top_k: int = Form(
        5,
        ge=1,
        le=20,
        description="Number of top ranked candidate matches to return",
    ),
    similarity_threshold: Optional[float] = Form(
        None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity filter",
    ),
    prompt_template: str = Form(
        DEFAULT_PROMPT_TEMPLATE,
        description="Prompt template string containing '{}' (e.g., 'a photo of a {}')",
    ),
    engine: OpenVocabEngine = Depends(get_open_vocab_engine),
) -> OpenVocabResponse:
    """Execute zero-shot open-vocabulary recognition using OpenCLIP ViT-B/32."""
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

    # 2. Validate and load image
    raw_bytes, filename, pil_img = await validate_and_read_image_upload(image, request=request)

    # 3. Execute CPU-bound inference in threadpool
    result = await run_in_threadpool(
        engine.recognize,
        image_input=pil_img,
        text_queries=queries,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        prompt_template=prompt_template,
    )

    if not result.get("success", False):
        raise ImageValidationError(
            message=result.get("error", "Open-vocabulary recognition failed during processing."),
            code=result.get("error_code", "INFERENCE_ERROR"),
            request_id=req_id,
        )

    # Attach original client filename and size
    if "image" in result:
        result["image"]["file_name"] = filename
        result["image"]["file_size_bytes"] = len(raw_bytes)

    return OpenVocabResponse.model_validate(result)
