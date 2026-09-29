"""Unified Open-Vocabulary Recognition Engine for Phase 6A."""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.phase3.image_validator import ValidationResult, validate_and_load_image
from src.phase6.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_TEXT_QUERIES,
    MIN_TEXT_QUERIES,
)
from src.phase6.open_vocab_classifier import OpenVocabClassifier

logger = logging.getLogger("phase6.engine")


class OpenVocabEngine:
    """Unified Open-Vocabulary Recognition Orchestrator."""

    def __init__(
        self,
        classifier: Optional[OpenVocabClassifier] = None,
        lazy_load: bool = False,
    ) -> None:
        """Initialize Open-Vocabulary Engine.

        Args:
            classifier: Pre-instantiated classifier instance or None.
            lazy_load: If True, defer model initialization until first call.
        """
        self._classifier = classifier
        if not lazy_load and self._classifier is None:
            self._classifier = OpenVocabClassifier()

    @property
    def classifier(self) -> OpenVocabClassifier:
        """Access pre-loaded classifier or lazily initialize."""
        if self._classifier is None:
            self._classifier = OpenVocabClassifier()
        return self._classifier

    def validate_queries(self, raw_queries: Union[List[str], str]) -> List[str]:
        """Sanitize and validate candidate text queries.

        Args:
            raw_queries: List of strings or comma-separated string.

        Returns:
            List of sanitized, valid concept strings.

        Raises:
            ValueError: If queries list is empty, exceeds 20 items, or contains invalid strings.
        """
        if isinstance(raw_queries, str):
            # Split comma-separated string
            items = [q.strip() for q in raw_queries.split(",") if q.strip()]
        elif isinstance(raw_queries, list):
            items = [str(q).strip() for q in raw_queries if str(q).strip()]
        else:
            raise ValueError("text_queries must be a list of strings or comma-separated string.")

        if len(items) < MIN_TEXT_QUERIES:
            raise ValueError("At least 1 candidate text query must be provided.")

        if len(items) > MAX_TEXT_QUERIES:
            raise ValueError(
                f"Maximum {MAX_TEXT_QUERIES} queries allowed per request. Received {len(items)}."
            )

        sanitized_queries: List[str] = []
        for q in items:
            # Strip control characters while preserving natural unicode
            clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", q)
            clean = re.sub(r"\s+", " ", clean).strip()

            if not clean:
                raise ValueError("Encountered empty query string after sanitization.")

            if len(clean) > MAX_QUERY_LENGTH:
                raise ValueError(
                    f"Query '{clean[:30]}...' exceeds maximum length of {MAX_QUERY_LENGTH} characters."
                )

            sanitized_queries.append(clean)

        return sanitized_queries

    def recognize_open_vocab(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, ValidationResult],
        text_queries: Union[List[str], str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Execute open-vocabulary zero-shot recognition on target image.

        Args:
            image_input: Filepath, PIL Image, Array, or ValidationResult.
            text_queries: List of candidate concept strings.
            prompt_template: Template string containing '{}'.
            min_similarity: Minimum cosine similarity cutoff.
            top_k: Top ranked items to return.

        Returns:
            Normalized dictionary adhering to the Phase 6A schema.
        """
        start_time = time.perf_counter()

        # Step 1: Validate Image Input using Phase 3 Validator
        if isinstance(image_input, ValidationResult):
            val_result = image_input
        else:
            val_result = validate_and_load_image(image_input)

        if not val_result.success:
            return {
                "success": False,
                "mode": "open_vocabulary",
                "error": val_result.error,
                "error_code": val_result.error_code,
            }

        meta = val_result.metadata
        assert meta is not None
        assert val_result.image is not None

        # Step 2: Validate and Sanitize Text Queries
        try:
            valid_queries = self.validate_queries(text_queries)
        except ValueError as e:
            return {
                "success": False,
                "mode": "open_vocabulary",
                "error": str(e),
                "error_code": "INVALID_TEXT_QUERIES",
            }

        # Step 3: Execute Zero-Shot Classification
        classification_output = self.classifier.classify_zero_shot(
            pil_image=val_result.image,
            text_queries=valid_queries,
            prompt_template=prompt_template,
            min_similarity=min_similarity,
            top_k=min(top_k, len(valid_queries)),
        )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 4: Construct Normalized Output
        image_info = {
            "file_name": meta.file_name,
            "file_path": meta.file_path,
            "width": meta.width,
            "height": meta.height,
            "channels": meta.channels,
            "format": meta.format,
            "aspect_ratio": round(meta.aspect_ratio, 3),
            "file_size_bytes": meta.file_size_bytes,
        }

        return {
            "success": True,
            "mode": "open_vocabulary",
            "image": image_info,
            "open_vocabulary": classification_output["open_vocabulary"],
            "metadata": {
                "engine": "AdvancedRecognitionEngine-v3-OpenVocab",
                "open_vocab_model": classification_output["open_vocabulary"]["model"],
                "total_inference_ms": round(total_elapsed_ms, 2),
            },
        }

    def recognize(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, ValidationResult],
        text_queries: Union[List[str], str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        similarity_threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Convenience alias for recognize_open_vocab."""
        cutoff = similarity_threshold if similarity_threshold is not None else min_similarity
        return self.recognize_open_vocab(
            image_input=image_input,
            text_queries=text_queries,
            prompt_template=prompt_template,
            min_similarity=cutoff,
            top_k=top_k,
        )

