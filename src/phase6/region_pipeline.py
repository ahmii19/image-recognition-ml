"""Production Two-Stage Grounded Region-Level Open-Vocabulary Recognition Pipeline."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.phase3.image_validator import ValidationResult, validate_and_load_image
from src.phase6.config import (
    DEFAULT_CROP_PADDING_PERCENT,
    DEFAULT_MAX_REGIONS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_REGION_DETECTION_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_ALLOWED_REGIONS,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
    MIN_TEXT_QUERIES,
    OPENCLIP_REGION_BATCH_SIZE,
)
from src.phase6.open_vocab_classifier import OpenCLIPClassifier
from src.phase6.owlvit.detector import OWLViTDetector

logger = logging.getLogger("phase6.region_pipeline")


def compute_padded_crop_box(
    box: Dict[str, int],
    image_width: int,
    image_height: int,
    padding_percent: int = DEFAULT_CROP_PADDING_PERCENT,
) -> Tuple[int, int, int, int]:
    """Calculate symmetrical padded bounding box clamped strictly to image boundaries.

    Args:
        box: Dict with 'x_min', 'y_min', 'x_max', 'y_max'.
        image_width: Total image width in pixels.
        image_height: Total image height in pixels.
        padding_percent: Symmetrical padding expansion percentage (e.g. 10 for 10%).

    Returns:
        Tuple of clamped coordinates: (x1, y1, x2, y2).
    """
    xmin = int(box["x_min"])
    ymin = int(box["y_min"])
    xmax = int(box["x_max"])
    ymax = int(box["y_max"])

    w = max(1, xmax - xmin)
    h = max(1, ymax - ymin)

    pad_ratio = max(0.0, float(padding_percent) / 100.0)
    pad_w = int(round(w * pad_ratio))
    pad_h = int(round(h * pad_ratio))

    crop_x1 = max(0, xmin - pad_w)
    crop_y1 = max(0, ymin - pad_h)
    crop_x2 = min(image_width, xmax + pad_w)
    crop_y2 = min(image_height, ymax + pad_h)

    # Ensure valid positive non-zero area (minimum 4x4 px)
    if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
        crop_x1 = max(0, min(xmin, image_width - 4))
        crop_y1 = max(0, min(ymin, image_height - 4))
        crop_x2 = min(image_width, max(xmax, crop_x1 + 4))
        crop_y2 = min(image_height, max(ymax, crop_y1 + 4))

    return crop_x1, crop_y1, crop_x2, crop_y2


class RegionRecognitionPipeline:
    """Production coordinator for Grounded Open-Vocabulary Object Recognition."""

    def __init__(
        self,
        detector: OWLViTDetector,
        classifier: OpenCLIPClassifier,
        batch_size: int = OPENCLIP_REGION_BATCH_SIZE,
    ) -> None:
        """Initialize pipeline with shared detector and classifier singletons.

        Args:
            detector: Singleton OWLViTDetector instance.
            classifier: Singleton OpenCLIPClassifier instance.
            batch_size: Maximum batch size for OpenCLIP vision encoding.
        """
        self.detector = detector
        self.classifier = classifier
        self.batch_size = max(1, batch_size)

    def process_image(
        self,
        image_input: Union[bytes, str, Image.Image],
        text_queries: Union[List[str], str],
        detection_threshold: float = DEFAULT_REGION_DETECTION_THRESHOLD,
        similarity_threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
        max_regions: int = DEFAULT_MAX_REGIONS,
        crop_padding_percent: int = DEFAULT_CROP_PADDING_PERCENT,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> Dict[str, Any]:
        """Execute end-to-end detection, context cropping, and batched semantic recognition.

        Args:
            image_input: Raw image bytes, file path, or PIL Image.
            text_queries: List of candidate concept queries or comma-separated string.
            detection_threshold: Cutoff score for OWL-ViT object localization.
            similarity_threshold: Optional minimum cosine cutoff for OpenCLIP refinement.
            top_k: Number of top candidate queries to retain per region.
            max_regions: Maximum candidate regions to analyze.
            crop_padding_percent: Context padding percentage (default 10%).
            prompt_template: Prompt template containing '{}' (default 'a photo of a {}').

        Returns:
            Complete structured response dictionary with per-region recognition and telemetry.
        """
        pipeline_start = time.perf_counter()

        # Step 1: Validate & Normalize Input Image
        validation: ValidationResult = validate_and_load_image(image_input)
        if not validation.success or validation.image is None:
            raise ValueError(f"Image validation failed: {validation.error or 'Invalid image'}")

        pil_image = validation.image
        img_w, img_h = pil_image.size

        # Step 2: Validate & Sanitize Candidate Queries
        sanitized_queries = self.detector.validate_queries(text_queries)

        # Enforce bounds
        bounded_max_regions = max(1, min(max_regions, MAX_ALLOWED_REGIONS))
        bounded_padding = max(0, min(crop_padding_percent, 50))
        bounded_top_k = max(1, min(top_k, len(sanitized_queries)))

        # Step 3: Stage 1 - OWL-ViT Open-Vocabulary Object Detection
        det_start = time.perf_counter()
        detection_result = self.detector.detect_objects(
            image_input=pil_image,
            text_queries=sanitized_queries,
            score_threshold=detection_threshold,
            top_k=bounded_max_regions * 2,  # detect broader pool then select best
            prompt_template=prompt_template,
        )
        det_latency_ms = (time.perf_counter() - det_start) * 1000.0

        raw_detections = detection_result.get("detections", [])

        # Step 4: Deterministic Region Selection (Score desc -> Area desc -> Index)
        def _region_priority(det: Dict[str, Any]) -> Tuple[float, int]:
            score = float(det.get("score", 0.0))
            box = det.get("box", {})
            area = int(box.get("width", 0)) * int(box.get("height", 0))
            return (score, area)

        sorted_detections = sorted(raw_detections, key=_region_priority, reverse=True)
        selected_detections = sorted_detections[:bounded_max_regions]

        # Step 5: Extract 10% Symmetrical Context Crops
        crop_start = time.perf_counter()
        valid_regions: List[Dict[str, Any]] = []
        cropped_images: List[Image.Image] = []

        for idx, det in enumerate(selected_detections):
            box = det["box"]
            # Validate bounding box bounds
            if box["width"] <= 0 or box["height"] <= 0:
                logger.warning(f"Skipping invalid zero-area detection box: {box}")
                continue

            c_x1, c_y1, c_x2, c_y2 = compute_padded_crop_box(
                box=box,
                image_width=img_w,
                image_height=img_h,
                padding_percent=bounded_padding,
            )

            crop_w = c_x2 - c_x1
            crop_h = c_y2 - c_y1
            if crop_w < 4 or crop_h < 4:
                continue

            crop_img = pil_image.crop((c_x1, c_y1, c_x2, c_y2))
            cropped_images.append(crop_img)

            valid_regions.append({
                "region_id": f"reg-{idx + 1:02d}",
                "detection": {
                    "label": det["query"],
                    "score": round(float(det["score"]), 4),
                    "prompt_used": det.get("prompt_used", f"a photo of a {det['query']}"),
                },
                "original_box": {
                    "x_min": box["x_min"],
                    "y_min": box["y_min"],
                    "x_max": box["x_max"],
                    "y_max": box["y_max"],
                    "width": box["width"],
                    "height": box["height"],
                },
                "padded_box": {
                    "x_min": c_x1,
                    "y_min": c_y1,
                    "x_max": c_x2,
                    "y_max": c_y2,
                    "width": crop_w,
                    "height": crop_h,
                },
                "normalized_box": {
                    "x_min": round(c_x1 / img_w, 4),
                    "y_min": round(c_y1 / img_h, 4),
                    "x_max": round(c_x2 / img_w, 4),
                    "y_max": round(c_y2 / img_h, 4),
                },
                "crop": {
                    "width": crop_w,
                    "height": crop_h,
                    "padding_percent": bounded_padding,
                },
            })

        crop_latency_ms = (time.perf_counter() - crop_start) * 1000.0

        # Step 6: Stage 2 - Batched OpenCLIP Semantic Recognition
        openclip_start = time.perf_counter()
        num_crops = len(cropped_images)

        if num_crops > 0:
            # 6A. Encode Candidate Text Queries ONCE (D x Q)
            tmpl = prompt_template if "{}" in prompt_template else f"{prompt_template} {{}}"
            formatted_prompts = [tmpl.format(q.strip()) for q in sanitized_queries]

            with torch.no_grad():
                text_tokens = self.classifier.tokenizer(formatted_prompts).to(self.classifier.device)
                text_features = self.classifier.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)  # (Q, D)

                # 6B. Batched Image Preprocessing & Encoding (Single Tensor Forward Pass)
                all_image_features = []
                for batch_offset in range(0, num_crops, self.batch_size):
                    batch_crops = cropped_images[batch_offset : batch_offset + self.batch_size]
                    tensors = [self.classifier.preprocess(c) for c in batch_crops]
                    batch_tensor = torch.stack(tensors, dim=0).to(self.classifier.device)
                    batch_feats = self.classifier.model.encode_image(batch_tensor)
                    batch_feats = F.normalize(batch_feats, dim=-1)  # (B, D)
                    all_image_features.append(batch_feats)

                stacked_image_features = torch.cat(all_image_features, dim=0)  # (N, D)

                # 6C. Vectorized Cosine Similarity Matrix: (N, D) @ (D, Q) -> (N, Q)
                similarity_matrix = (stacked_image_features @ text_features.T).cpu().numpy()

            # 6D. Map Ranked Semantic Alignments to Each Region
            for r_idx, reg in enumerate(valid_regions):
                scores = similarity_matrix[r_idx]
                ranked_indices = np.argsort(scores)[::-1]

                ranked_items: List[Dict[str, Any]] = []
                for rank_pos, q_idx in enumerate(ranked_indices[:bounded_top_k], start=1):
                    s_val = round(float(scores[q_idx]), 4)
                    if similarity_threshold is not None and s_val < similarity_threshold:
                        continue
                    ranked_items.append({
                        "query": sanitized_queries[q_idx],
                        "similarity_score": s_val,
                        "rank": rank_pos,
                    })

                top_match = ranked_items[0] if ranked_items else None
                reg["recognition"] = {
                    "top_match": top_match,
                    "rankings": ranked_items,
                    "total_candidates": len(sanitized_queries),
                }

        openclip_latency_ms = (time.perf_counter() - openclip_start) * 1000.0
        total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

        return {
            "success": True,
            "mode": "region_recognition",
            "image": {
                "file_name": validation.metadata.file_name if validation.metadata else "image.jpg",
                "width": img_w,
                "height": img_h,
                "channels": 3,
                "format": validation.metadata.format if validation.metadata else "JPEG",
                "aspect_ratio": round(img_w / max(1, img_h), 4),
                "file_size_bytes": validation.metadata.file_size_bytes if validation.metadata else 0,
            },
            "summary": {
                "total_queries": len(sanitized_queries),
                "detected_regions_count": len(raw_detections),
                "analyzed_regions_count": len(valid_regions),
                "detection_threshold": detection_threshold,
                "crop_padding_percent": bounded_padding,
                "batch_size_used": min(self.batch_size, max(1, num_crops)),
                "prompt_template": prompt_template,
            },
            "models": {
                "detector": "OWL-ViT base patch32",
                "recognizer": f"OpenCLIP-{self.classifier.model_name}",
                "detector_parameters": getattr(self.detector, "param_count", 153231879),
                "device": str(self.classifier.device),
            },
            "regions": valid_regions,
            "timing": {
                "owlvit_detection_ms": round(det_latency_ms, 2),
                "crop_preparation_ms": round(crop_latency_ms, 2),
                "openclip_batch_ms": round(openclip_latency_ms, 2),
                "total_pipeline_ms": round(total_pipeline_ms, 2),
            },
        }
