"""Production-oriented Image Classification Engine for Phase 3."""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_decode, preprocess_input as mobilenet_preprocess

from src.phase3.config import DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD
from src.phase3.image_validator import ValidationResult, validate_and_load_image


class Phase3Classifier:
    """ImageNet-1K / Multi-Class Image Classification Engine."""

    def __init__(
        self,
        model_name: str = "MobileNetV2",
        confidence_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize the ImageNet classification model.

        Args:
            model_name: Backbone architecture ("MobileNetV2" or "EfficientNetB0").
            confidence_threshold: Minimum confidence threshold for certainty.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.input_shape = (224, 224)

        print(f"[INFO] Initializing Phase 3 Classifier ({self.model_name}) on ImageNet-1K...")
        if self.model_name.lower() in ("mobilenetv2", "mobilenet_v2"):
            self.model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
            self._preprocess_fn = mobilenet_preprocess
        else:
            self.model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=True)
            self._preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

        # Warmup forward pass
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print(f"[INFO] Phase 3 Classifier ({self.model_name}) ready.")

    def preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        """Resize image with aspect-preserving center crop and apply model-specific normalization."""
        target_w, target_h = self.input_shape
        orig_w, orig_h = pil_img.size

        # Compute aspect-preserving scale
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop to exact target shape
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        cropped = resized.crop((left, top, left + target_w, top + target_h))

        arr = np.array(cropped, dtype=np.float32)
        batch = np.expand_dims(arr, axis=0)
        return self._preprocess_fn(batch)

    def classify(
        self,
        image_input: Union[str, Image.Image, np.ndarray, ValidationResult],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classify an image input and return top-k predictions with confidence analysis.

        Args:
            image_input: Filepath, PIL Image, Numpy Array, or pre-validated ValidationResult.
            top_k: Number of ranked candidate classes to return.
            threshold: Optional custom confidence threshold override.

        Returns:
            Structured dictionary matching Phase 3 schema.
        """
        conf_thresh = threshold if threshold is not None else self.confidence_threshold

        # Step 1: Validate & load image
        if isinstance(image_input, ValidationResult):
            val_res = image_input
        else:
            val_res = validate_and_load_image(image_input)

        if not val_res.success:
            return {
                "success": False,
                "mode": "classification",
                "error": val_res.error,
                "error_code": val_res.error_code,
            }

        pil_img = val_res.image
        assert pil_img is not None

        # Step 2: Preprocess
        prep_start = time.perf_counter()
        input_tensor = self.preprocess_image(pil_img)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # Step 3: Inference
        inf_start = time.perf_counter()
        raw_probs = self.model.predict(input_tensor, verbose=0)[0]
        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        # Step 4: Decode top-k
        decoded = mobilenet_decode(np.expand_dims(raw_probs, axis=0), top=top_k)[0]
        predictions: List[Dict[str, Any]] = []

        for _, raw_label, prob in decoded:
            clean_label = raw_label.replace("_", " ")
            conf_val = float(prob)
            predictions.append(
                {
                    "label": clean_label,
                    "confidence": round(conf_val, 4),
                    "confidence_percent": round(conf_val * 100.0, 2),
                }
            )

        top1 = predictions[0]
        is_confident = top1["confidence"] >= conf_thresh
        status = "confident" if is_confident else "low_confidence"

        return {
            "success": True,
            "mode": "classification",
            "status": status,
            "top1": top1,
            "top5": predictions[:5],
            "predictions": predictions,
            "confidence_threshold": conf_thresh,
            "is_confident": is_confident,
            "inference_ms": round(inf_ms, 2),
            "preprocessing_ms": round(prep_ms, 2),
            "total_ms": round(prep_ms + inf_ms, 2),
            "model": self.model_name,
            "input_dimensions": f"{self.input_shape[0]}x{self.input_shape[1]}x3",
        }
