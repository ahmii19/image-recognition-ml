"""Phase 6A: Open-Vocabulary Vision-Language Recognition Engine using OpenCLIP ViT-B/32."""

from src.phase6.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PROMPT_TEMPLATE,
    MAX_QUERY_LENGTH,
    MAX_TEXT_QUERIES,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_NUM_THREADS,
    OPENCLIP_PRETRAINED,
)
from src.phase6.engine import OpenVocabEngine
from src.phase6.open_vocab_classifier import OpenVocabClassifier

__all__ = [
    "OPENCLIP_MODEL_NAME",
    "OPENCLIP_PRETRAINED",
    "OPENCLIP_NUM_THREADS",
    "DEFAULT_PROMPT_TEMPLATE",
    "MAX_TEXT_QUERIES",
    "MAX_QUERY_LENGTH",
    "DEFAULT_MIN_SIMILARITY",
    "OpenVocabClassifier",
    "OpenVocabEngine",
]
