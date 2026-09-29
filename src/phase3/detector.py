"""Modern Pre-Trained Object Detection Engine for Phase 3."""

import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import tensorflow as tf

from src.phase3.config import (
    COCO_LABELS,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DETECTOR_LOCAL_PATH,
    DETECTOR_MODEL_ARCHIVE,
    DETECTOR_MODEL_NAME,
    DETECTOR_MODEL_URL,
    PHASE3_DATA_DIR,
    PHASE3_MODELS_DIR,
    ensure_phase3_directories,
)
from src.phase3.image_validator import ValidationResult, validate_and_load_image


class Phase3ObjectDetector:
    """COCO Pre-Trained Object Detection Engine (SSD MobileNet V2)."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_DETECTION_IOU_THRESHOLD,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the object detector.

        Args:
            confidence_threshold: Minimum score threshold for positive detection.
            iou_threshold: Overlap threshold for NMS.
            model_path: Optional path to custom SavedModel directory.
        """
        ensure_phase3_directories()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_name = "SSD-MobileNetV2-COCO"
        self.model_path = Path(model_path) if model_path else self._resolve_or_download_model()

        print(f"[INFO] Loading Phase 3 Object Detector from: {self.model_path}...")
        self.detector = tf.saved_model.load(str(self.model_path))

        # Warmup forward pass
        dummy = tf.zeros((1, 320, 320, 3), dtype=tf.uint8)
        _ = self.detector(dummy)
        print(f"[INFO] Phase 3 Object Detector ({self.model_name}) initialized successfully.")

    def _resolve_or_download_model(self) -> Path:
        """Ensure detector weights are available locally, unpacking if necessary."""
        # Check standard unpacked path
        candidates = [
            DETECTOR_LOCAL_PATH,
            PHASE3_MODELS_DIR / DETECTOR_MODEL_NAME / "saved_model",
            PHASE3_DATA_DIR / "models" / DETECTOR_MODEL_NAME / "saved_model",
            PHASE3_DATA_DIR / "models" / "saved_model",
        ]

        for cand in candidates:
            if cand.exists() and (cand / "saved_model.pb").exists():
                return cand

        # Check for downloaded tarball
        tarball_paths = [
            PHASE3_DATA_DIR / "models" / DETECTOR_MODEL_ARCHIVE,
            PHASE3_MODELS_DIR / DETECTOR_MODEL_ARCHIVE,
        ]

        for tb in tarball_paths:
            if tb.exists():
                print(f"[INFO] Unpacking detector archive: {tb}...")
                with tarfile.open(tb, "r:gz") as tar:
                    tar.extractall(path=PHASE3_MODELS_DIR)
                extracted_saved_model = PHASE3_MODELS_DIR / DETECTOR_MODEL_NAME / "saved_model"
                if extracted_saved_model.exists():
                    return extracted_saved_model

        # Download if not present
        print(f"[INFO] Downloading SSD MobileNet V2 from {DETECTOR_MODEL_URL}...")
        downloaded = tf.keras.utils.get_file(
            fname=DETECTOR_MODEL_ARCHIVE,
            origin=DETECTOR_MODEL_URL,
            untar=True,
            cache_dir=str(PHASE3_DATA_DIR),
            cache_subdir="models",
        )
        downloaded_dir = Path(downloaded)
        extracted = downloaded_dir / "saved_model" if (downloaded_dir / "saved_model").exists() else downloaded_dir
        return extracted

    def detect(
        self,
        image_input: Union[str, Image.Image, np.ndarray, ValidationResult],
        threshold: Optional[float] = None,
        max_detections: int = 20,
    ) -> Dict[str, Any]:
        """Detect objects in the image, returning bounding boxes and class probabilities.

        Args:
            image_input: Target image as path, PIL Image, array, or pre-validated result.
            threshold: Confidence threshold override.
            max_detections: Upper limit of detected items to return.

        Returns:
            Structured detection dictionary.
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
                "mode": "detection",
                "error": val_res.error,
                "error_code": val_res.error_code,
            }

        pil_img = val_res.image
        assert pil_img is not None
        orig_w, orig_h = pil_img.size

        # Step 2: Convert to tensor (uint8 [0, 255] RGB)
        prep_start = time.perf_counter()
        img_np = np.array(pil_img, dtype=np.uint8)
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, axis=0), dtype=tf.uint8)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # Step 3: Run Object Detection Inference
        inf_start = time.perf_counter()
        raw_outputs = self.detector(input_tensor)
        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        # Step 4: Extract predictions
        boxes = raw_outputs["detection_boxes"][0].numpy()      # [ymin, xmin, ymax, xmax] in [0, 1]
        classes = raw_outputs["detection_classes"][0].numpy()  # float class IDs
        scores = raw_outputs["detection_scores"][0].numpy()    # float scores [0, 1]

        detected_objects: List[Dict[str, Any]] = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < conf_thresh:
                continue

            class_id = int(classes[i])
            label = COCO_LABELS.get(class_id, f"object_{class_id}")
            ymin, xmin, ymax, xmax = boxes[i]

            # Scale normalized coordinates to original image pixel coordinates
            x1 = int(round(float(xmin) * orig_w))
            y1 = int(round(float(ymin) * orig_h))
            x2 = int(round(float(xmax) * orig_w))
            y2 = int(round(float(ymax) * orig_h))

            # Clamp coordinates
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(x1 + 1, min(orig_w, x2))
            y2 = max(y1 + 1, min(orig_h, y2))
            box_w = x2 - x1
            box_h = y2 - y1

            detected_objects.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "confidence": round(score, 4),
                    "confidence_percent": round(score * 100.0, 2),
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": box_w,
                        "height": box_h,
                        "normalized": [
                            round(float(ymin), 4),
                            round(float(xmin), 4),
                            round(float(ymax), 4),
                            round(float(xmax), 4),
                        ],
                    },
                }
            )

            if len(detected_objects) >= max_detections:
                break

        status = "objects_detected" if len(detected_objects) > 0 else "no_confident_detections"

        return {
            "success": True,
            "mode": "detection",
            "status": status,
            "count": len(detected_objects),
            "objects": detected_objects,
            "confidence_threshold": conf_thresh,
            "inference_ms": round(inf_ms, 2),
            "preprocessing_ms": round(prep_ms, 2),
            "total_ms": round(prep_ms + inf_ms, 2),
            "model": self.model_name,
            "image_dimensions": {"width": orig_w, "height": orig_h},
        }
