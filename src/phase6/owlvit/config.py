"""Configuration and environment settings for OWL-ViT Open-Vocabulary Object Detection."""

import os
from typing import Final

# Model Identifiers
OWL_VIT_MODEL_ID: Final[str] = os.getenv("OWL_VIT_MODEL_ID", "google/owlvit-base-patch32")
OWL_VIT_DEVICE: Final[str] = os.getenv("OWL_VIT_DEVICE", os.getenv("ML_DEVICE", "auto"))
OWL_VIT_TORCH_THREADS: Final[int] = int(os.getenv("OWL_VIT_TORCH_THREADS", "4"))

# Query & Processing Limits
MIN_TEXT_QUERIES: Final[int] = 1
MAX_TEXT_QUERIES: Final[int] = int(os.getenv("OWL_VIT_MAX_QUERIES", "20"))
MAX_TEXT_QUERY_LENGTH: Final[int] = int(os.getenv("OWL_VIT_MAX_QUERY_LENGTH", "128"))
MAX_UPLOAD_SIZE_MB: Final[int] = int(os.getenv("OWL_VIT_MAX_IMAGE_MB", "10"))

# Prompt Normalization Defaults
DEFAULT_PROMPT_TEMPLATE: Final[str] = os.getenv("OWL_VIT_PROMPT_TEMPLATE", "a photo of a {}")
DEFAULT_SCORE_THRESHOLD: Final[float] = float(os.getenv("OWL_VIT_SCORE_THRESHOLD", "0.10"))
DEFAULT_TOP_K: Final[int] = int(os.getenv("OWL_VIT_DEFAULT_TOP_K", "20"))

# Lifecycle & Memory Management
AUTO_UNLOAD_ENABLED: Final[bool] = os.getenv("OWL_VIT_AUTO_UNLOAD", "false").lower() in ("true", "1", "yes")
IDLE_UNLOAD_SECONDS: Final[int] = int(os.getenv("OWL_VIT_IDLE_UNLOAD_SECONDS", "300"))
