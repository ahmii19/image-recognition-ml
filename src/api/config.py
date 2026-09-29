"""Environment and Server Configuration for FastAPI Serving Layer."""

import os
from typing import List
from src.phase3.config import SUPPORTED_IMAGE_EXTENSIONS

# Server Configuration
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_TITLE: str = "Image Recognition & Object Detection API"
API_DESCRIPTION: str = (
    "Production-grade FastAPI serving layer for ML Image Recognition, "
    "featuring MobileNetV2 ImageNet-1K classification and SSD-MobileNetV2 COCO object detection."
)
API_VERSION: str = "1.0.0"
API_V1_STR: str = "/api/v1"

# Upload Limits & Validation
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS: set = {ext.lstrip(".").lower() for ext in SUPPORTED_IMAGE_EXTENSIONS}

# CORS Configuration
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

# Logging & Telemetry
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
