"""FastAPI Dependencies for Model Injection and Request Validation."""

import io
from pathlib import Path
from typing import Optional
from fastapi import File, Request, UploadFile
from PIL import Image

from src.api.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from src.api.errors import (
    ImageValidationError,
    ModelUnavailableError,
    PayloadTooLargeError,
)
from src.phase3.engine import AdvancedRecognitionEngine
from src.phase6.engine import OpenVocabEngine
from src.phase6.owlvit.detector import OWLViTDetector
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager


def get_recognition_engine(request: Request) -> AdvancedRecognitionEngine:
    """Retrieve pre-loaded singleton AdvancedRecognitionEngine from application state."""
    engine: Optional[AdvancedRecognitionEngine] = getattr(request.app.state, "engine", None)
    if engine is None:
        raise ModelUnavailableError("ML Recognition Engine is not initialized or unavailable.")
    return engine


def get_open_vocab_engine(request: Request) -> OpenVocabEngine:
    """Retrieve pre-loaded singleton OpenVocabEngine from application state."""
    engine: Optional[OpenVocabEngine] = getattr(request.app.state, "open_vocab_engine", None)
    if engine is None:
        raise ModelUnavailableError("Open-Vocabulary Recognition Engine is not initialized or unavailable.")
    return engine


def get_owlvit_detector() -> OWLViTDetector:
    """Retrieve thread-safe lazy-loaded OWLViTDetector singleton.

    Does NOT load model on startup; loads on-demand on first request.
    """
    try:
        return OWLViTLifecycleManager.get_detector()
    except Exception as e:
        raise ModelUnavailableError(
            f"OWL-ViT Open-Vocabulary Object Detector could not be initialized: {str(e)}"
        )


async def validate_and_read_image_upload(
    image: UploadFile = File(..., description="Uploaded image file (JPG, PNG, WEBP, BMP)"),
    request: Request = None,
) -> tuple[bytes, str, Image.Image]:
    """Read, stream-validate file size, and decode image without writing insecure disk files.

    Args:
        image: Multipart file upload.
        request: FastAPI Request instance for correlation ID.

    Returns:
        Tuple of (raw_bytes, sanitized_filename, PIL.Image).

    Raises:
        PayloadTooLargeError: If file exceeds MAX_UPLOAD_SIZE_MB.
        ImageValidationError: If file is missing, empty, or has unsupported extension.
    """
    req_id = getattr(request.state, "request_id", None) if request else None

    # 1. Filename & Extension Check
    filename = Path(image.filename or "uploaded_image.jpg").name
    suffix = Path(filename).suffix.lower().lstrip(".")

    if suffix and suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '.{suffix}'. Supported formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}.",
            code="UNSUPPORTED_FORMAT",
            request_id=req_id,
        )

    # 2. Stream-read with hard byte limit to prevent memory exhaustion (DoS)
    chunk_size = 64 * 1024  # 64 KB chunks
    total_bytes = bytearray()

    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        total_bytes.extend(chunk)
        if len(total_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise PayloadTooLargeError(
                f"Uploaded image exceeds maximum allowable limit of {MAX_UPLOAD_SIZE_MB}MB.",
                request_id=req_id,
            )

    raw_data = bytes(total_bytes)

    if len(raw_data) == 0:
        raise ImageValidationError(
            "Uploaded file is empty (0 bytes).",
            code="EMPTY_FILE",
            request_id=req_id,
        )

    # 3. Basic PIL Image verification
    try:
        pil_img = Image.open(io.BytesIO(raw_data))
        pil_img.verify()
        # Re-open after verify
        pil_img = Image.open(io.BytesIO(raw_data))
        pil_img.load()
    except Exception as e:
        raise ImageValidationError(
            f"Uploaded file is corrupted or cannot be decoded as an image: {str(e)}",
            code="CORRUPTED_IMAGE",
            request_id=req_id,
        )

    return raw_data, filename, pil_img
