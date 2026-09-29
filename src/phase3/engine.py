"""Unified Advanced Recognition Engine for Phase 3."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from src.phase3.classifier import Phase3Classifier
from src.phase3.config import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
    PHASE3_DETECTIONS_DIR,
    ensure_phase3_directories,
)
from src.phase3.detector import Phase3ObjectDetector
from src.phase3.image_validator import ValidationResult, validate_and_load_image
from src.phase3.visualizer import render_detections


class AdvancedRecognitionEngine:
    """Unified ML Image Recognition Engine integrating Classification and Object Detection."""

    def __init__(
        self,
        classifier: Optional[Phase3Classifier] = None,
        detector: Optional[Phase3ObjectDetector] = None,
        lazy_load: bool = False,
    ) -> None:
        """Initialize the unified recognition engine.

        Args:
            classifier: Optional pre-instantiated classifier.
            detector: Optional pre-instantiated detector.
            lazy_load: If True, defer model loading until first inference call.
        """
        ensure_phase3_directories()
        self._classifier = classifier
        self._detector = detector

        if not lazy_load:
            self._ensure_models_loaded()

    def _ensure_models_loaded(self) -> None:
        """Load underlying neural network models if not already active."""
        if self._classifier is None:
            self._classifier = Phase3Classifier()
        if self._detector is None:
            self._detector = Phase3ObjectDetector()

    @property
    def classifier(self) -> Phase3Classifier:
        if self._classifier is None:
            self._classifier = Phase3Classifier()
        return self._classifier

    @property
    def detector(self) -> Phase3ObjectDetector:
        if self._detector is None:
            self._detector = Phase3ObjectDetector()
        return self._detector

    def recognize(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        mode: str = "all",
        classification_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        detection_threshold: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        render_visualization: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Execute high-level image recognition on target image.

        Args:
            image_input: Filepath, PIL Image, or Numpy Array.
            mode: Recognition mode ("all", "classification", or "detection").
            classification_threshold: Confidence cutoff for classification certainty.
            detection_threshold: Confidence cutoff for object detection boxes.
            render_visualization: If True, generates visual bounding box image in outputs/.
            top_k: Number of candidate classes to return.

        Returns:
            Normalized, JSON-serializable recognition result dictionary.
        """
        mode_lower = mode.lower()
        if mode_lower not in ("all", "classification", "detection"):
            return {
                "success": False,
                "error": f"Invalid mode '{mode}'. Choose 'all', 'classification', or 'detection'.",
                "error_code": "INVALID_MODE",
            }

        start_time = time.perf_counter()

        # Step 1: Validate Image Input
        val_result: ValidationResult = validate_and_load_image(image_input)
        if not val_result.success:
            return {
                "success": False,
                "error": val_result.error,
                "error_code": val_result.error_code,
            }

        meta = val_result.metadata
        assert meta is not None
        assert val_result.image is not None

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

        classification_result: Optional[Dict[str, Any]] = None
        detection_result: Optional[Dict[str, Any]] = None
        rendered_path: Optional[str] = None

        # Step 2: Perform Classification if requested
        if mode_lower in ("all", "classification"):
            classification_result = self.classifier.classify(
                val_result,
                top_k=top_k,
                threshold=classification_threshold,
            )

        # Step 3: Perform Object Detection if requested
        if mode_lower in ("all", "detection"):
            detection_result = self.detector.detect(
                val_result,
                threshold=detection_threshold,
            )

            # Step 4: Render visualization if detections exist
            if render_visualization and detection_result.get("success", False):
                stem = Path(meta.file_name).stem if meta.file_name else "image"
                out_name = f"{stem}_detected.png"
                dest = PHASE3_DETECTIONS_DIR / out_name
                saved_dest = render_detections(
                    val_result.image,
                    detection_result.get("objects", []),
                    save_path=dest,
                )
                rendered_path = str(saved_dest)
                detection_result["rendered_image_path"] = rendered_path

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Construct Unified Output
        output: Dict[str, Any] = {
            "success": True,
            "mode": mode_lower,
            "image": image_info,
            "metadata": {
                "engine": "AdvancedRecognitionEngine-v3",
                "classifier_model": self.classifier.model_name if mode_lower in ("all", "classification") else None,
                "detector_model": self.detector.model_name if mode_lower in ("all", "detection") else None,
                "classification_threshold": classification_threshold,
                "detection_threshold": detection_threshold,
                "total_inference_ms": round(total_elapsed_ms, 2),
            },
        }

        if classification_result is not None:
            output["classification"] = classification_result

        if detection_result is not None:
            output["detection"] = detection_result

        return output
