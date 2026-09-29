"""Configuration and Hyperparameters for Phase 6A Open-Vocabulary Recognition."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
PHASE6_CACHE_DIR: Path = PROJECT_ROOT / "models" / "phase6"

# Model Selection
# ViT-B/32 trained on LAION-2B offers high zero-shot transfer accuracy
# with a fast, lightweight CPU footprint (~350MB weights, ~450MB RAM).
OPENCLIP_MODEL_NAME: str = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
OPENCLIP_PRETRAINED: str = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")

# CPU Threading
# 4 threads matches the 4 physical CPU cores of the host Intel i5 processor
# without oversubscribing context switches.
OPENCLIP_NUM_THREADS: int = int(os.getenv("OPENCLIP_NUM_THREADS", "4"))
OPENCLIP_DEVICE: str = os.getenv("OPENCLIP_DEVICE", os.getenv("ML_DEVICE", "auto"))

# Query Constraints & Prompt Templates
DEFAULT_PROMPT_TEMPLATE: str = os.getenv("OPENCLIP_PROMPT_TEMPLATE", "a photo of a {}")
MAX_TEXT_QUERIES: int = int(os.getenv("OPENCLIP_MAX_QUERIES", "20"))
MIN_TEXT_QUERIES: int = 1
MAX_QUERY_LENGTH: int = int(os.getenv("OPENCLIP_MAX_QUERY_LENGTH", "128"))
MAX_TEXT_QUERY_LENGTH: int = MAX_QUERY_LENGTH

# Similarity & Ranking Defaults
# Note: Cosine similarity is in [-1.0, 1.0]. A heuristic cutoff of 0.20
# filters out clearly irrelevant negative noise without forcing false matches.
DEFAULT_MIN_SIMILARITY: float = float(os.getenv("OPENCLIP_MIN_SIMILARITY", "0.20"))
DEFAULT_TOP_K: int = int(os.getenv("OPENCLIP_TOP_K", "5"))
MAX_TOP_K: int = 20

# Phase 6C Region-Level Recognition Constants
DEFAULT_CROP_PADDING_PERCENT: int = int(os.getenv("PHASE6C_CROP_PADDING_PERCENT", "10"))
DEFAULT_MAX_REGIONS: int = int(os.getenv("PHASE6C_MAX_REGIONS", "20"))
MAX_ALLOWED_REGIONS: int = 50
OPENCLIP_REGION_BATCH_SIZE: int = int(os.getenv("PHASE6C_REGION_BATCH_SIZE", "20"))
DEFAULT_REGION_DETECTION_THRESHOLD: float = 0.10


def ensure_phase6_directories() -> None:
    """Ensure directory structure exists for Phase 6 cache."""
    PHASE6_CACHE_DIR.mkdir(parents=True, exist_ok=True)

