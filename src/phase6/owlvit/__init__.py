"""Phase 6B Open-Vocabulary Object Detection Module (OWL-ViT)."""

from src.phase6.owlvit.config import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
    OWL_VIT_DEVICE,
    OWL_VIT_MODEL_ID,
    OWL_VIT_TORCH_THREADS,
)
from src.phase6.owlvit.detector import OWLViTDetector
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager

__all__ = [
    "OWLViTDetector",
    "OWLViTLifecycleManager",
    "OWL_VIT_MODEL_ID",
    "OWL_VIT_DEVICE",
    "OWL_VIT_TORCH_THREADS",
    "MAX_TEXT_QUERIES",
    "MAX_TEXT_QUERY_LENGTH",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_TOP_K",
]
