"""Core Open-Vocabulary Object Detector using Google OWL-ViT."""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from src.device import resolve_device
from src.phase3.image_validator import ValidationResult, validate_and_load_image
from src.phase6.owlvit.config import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
    MIN_TEXT_QUERIES,
    OWL_VIT_DEVICE,
    OWL_VIT_MODEL_ID,
    OWL_VIT_TORCH_THREADS,
)

logger = logging.getLogger("phase6.owlvit.detector")


class OWLViTDetector:
    """Production Open-Vocabulary Object Detector powered by OWL-ViT supporting CPU and CUDA."""

    def __init__(
        self,
        model_id: str = OWL_VIT_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        torch_threads: int = OWL_VIT_TORCH_THREADS,
    ) -> None:
        """Initialize and instantiate the OWL-ViT model and processor.

        Args:
            model_id: HuggingFace model checkpoint identifier.
            device: Target execution device ('cpu', 'cuda', or 'auto' / None).
            torch_threads: Number of PyTorch CPU intra-op threads.
        """
        self.model_id = model_id
        self.device = str(resolve_device(device if device is not None else OWL_VIT_DEVICE))
        self.torch_threads = torch_threads

        logger.info(
            f"Loading OWL-ViT Detector from '{self.model_id}' on device '{self.device}'..."
        )
        if self.device == "cpu" and self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)

        start_time = time.perf_counter()
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.processor: OwlViTProcessor = OwlViTProcessor.from_pretrained(self.model_id)
        self.model: OwlViTForObjectDetection = OwlViTForObjectDetection.from_pretrained(
            self.model_id
        )
        self.model.to(self.device)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())
        elapsed_s = time.perf_counter() - start_time
        logger.info(
            f"OWL-ViT Detector successfully initialized in {elapsed_s:.2f}s "
            f"({self.param_count:,} parameters)."
        )

    def validate_queries(self, raw_queries: Union[List[str], str]) -> List[str]:
        """Validate, sanitize, and deduplicate natural-language candidate queries.

        Args:
            raw_queries: List of strings or comma-separated query string.

        Returns:
            List of sanitized, unique query strings.

        Raises:
            ValueError: If query count or lengths exceed configured bounds.
        """
        if isinstance(raw_queries, str):
            items = [q.strip() for q in raw_queries.split(",") if q.strip()]
        elif isinstance(raw_queries, list):
            items = [str(q).strip() for q in raw_queries if str(q).strip()]
        else:
            raise ValueError("text_queries must be a list of strings or a comma-separated string.")

        if len(items) < MIN_TEXT_QUERIES:
            raise ValueError("At least 1 candidate text query must be provided.")

        if len(items) > MAX_TEXT_QUERIES:
            raise ValueError(
                f"Maximum of {MAX_TEXT_QUERIES} queries allowed per request. Received {len(items)}."
            )

        sanitized: List[str] = []
        seen = set()
        for q in items:
            # Strip control characters while preserving Unicode letters
            clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", q)
            clean = re.sub(r"\s+", " ", clean).strip()

            if not clean:
                raise ValueError("Encountered an empty query after whitespace sanitization.")

            if len(clean) > MAX_TEXT_QUERY_LENGTH:
                raise ValueError(
                    f"Query '{clean[:25]}...' exceeds maximum length of {MAX_TEXT_QUERY_LENGTH} characters."
                )

            # Preserve ordering while eliminating case-insensitive duplicates
            lower_q = clean.lower()
            if lower_q not in seen:
                seen.add(lower_q)
                sanitized.append(clean)

        return sanitized

    def format_prompt(self, query: str, template: Optional[str] = None) -> str:
        """Format candidate query with ensembled prompt template for high detection recall.

        Args:
            query: Raw user concept (e.g., 'dog').
            template: Optional prompt template containing '{}'.

        Returns:
            Formatted prompt string.
        """
        tmpl = template if template and "{}" in template else DEFAULT_PROMPT_TEMPLATE
        try:
            return tmpl.format(query.strip())
        except Exception:
            return DEFAULT_PROMPT_TEMPLATE.format(query.strip())

    @torch.no_grad()
    def detect_objects(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, ValidationResult],
        text_queries: Union[List[str], str],
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> Dict[str, Any]:
        """Execute open-vocabulary object localization and bounding box prediction.

        Args:
            image_input: Image filepath, PIL Image, NumPy array, or ValidationResult.
            text_queries: Candidate concept labels.
            score_threshold: Minimum detection confidence cutoff [0.0, 1.0].
            top_k: Maximum total detections to return across all queries.
            prompt_template: Prompt template containing '{}' (e.g. 'a photo of a {}').

        Returns:
            Structured dictionary matching the Phase 6B Open-Vocabulary Detection schema.
        """
        start_time = time.perf_counter()

        # Step 1: Validate Image Input via Phase 3 Image Validator
        if isinstance(image_input, ValidationResult):
            val_result = image_input
        else:
            val_result = validate_and_load_image(image_input)

        if not val_result.success:
            return {
                "success": False,
                "mode": "open_vocabulary_detection",
                "error": val_result.error,
                "error_code": val_result.error_code,
            }

        meta = val_result.metadata
        assert meta is not None
        assert val_result.image is not None
        pil_image: Image.Image = val_result.image
        img_width, img_height = pil_image.size

        # Step 2: Validate and Sanitize Text Queries
        try:
            valid_queries = self.validate_queries(text_queries)
        except ValueError as e:
            return {
                "success": False,
                "mode": "open_vocabulary_detection",
                "error": str(e),
                "error_code": "INVALID_TEXT_QUERIES",
            }

        # Step 3: Format Prompts
        formatted_prompts = [self.format_prompt(q, prompt_template) for q in valid_queries]

        # Step 4: Tokenize & Preprocess Inputs
        prep_start = time.perf_counter()
        inputs = self.processor(
            text=[formatted_prompts],
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # Step 5: Model Forward Pass
        inf_start = time.perf_counter()
        outputs = self.model(**inputs)
        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        # Step 6: Post-Process Bounding Boxes and Align with Queries
        target_sizes = torch.tensor([[img_height, img_width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=float(score_threshold),
            target_sizes=target_sizes,
            text_labels=[formatted_prompts],
        )

        res = results[0]
        raw_boxes = res.get("boxes", torch.empty((0, 4))).cpu().tolist()
        raw_scores = res.get("scores", torch.empty(0)).cpu().tolist()
        raw_labels = res.get("text_labels", res.get("labels", []))

        # Group detections by original user query
        query_groups: Dict[str, List[Dict[str, Any]]] = {q: [] for q in valid_queries}
        all_detections: List[Dict[str, Any]] = []

        for idx, (box, score, label) in enumerate(zip(raw_boxes, raw_scores, raw_labels)):
            # Label index or string mapping
            if isinstance(label, int) and 0 <= label < len(valid_queries):
                matched_query = valid_queries[label]
                matched_prompt = formatted_prompts[label]
            elif isinstance(label, str):
                # Reverse match formatted prompt to valid query
                matched_query = label
                matched_prompt = label
                for orig_q, prompt_q in zip(valid_queries, formatted_prompts):
                    if label == prompt_q or label == orig_q:
                        matched_query = orig_q
                        matched_prompt = prompt_q
                        break
            else:
                matched_query = valid_queries[0]
                matched_prompt = formatted_prompts[0]

            x_min, y_min, x_max, y_max = [float(v) for v in box]

            # Clamp coordinates to image boundaries
            x_min_c = max(0.0, min(float(img_width), x_min))
            y_min_c = max(0.0, min(float(img_height), y_min))
            x_max_c = max(0.0, min(float(img_width), x_max))
            y_max_c = max(0.0, min(float(img_height), y_max))

            width_px = max(1.0, x_max_c - x_min_c)
            height_px = max(1.0, y_max_c - y_min_c)

            norm_xmin = round(x_min_c / img_width, 4)
            norm_ymin = round(y_min_c / img_height, 4)
            norm_xmax = round(x_max_c / img_width, 4)
            norm_ymax = round(y_max_c / img_height, 4)

            det_item = {
                "detection_id": f"det-{idx + 1}",
                "query": matched_query,
                "prompt_used": matched_prompt,
                "score": round(float(score), 4),
                "box": {
                    "x_min": int(round(x_min_c)),
                    "y_min": int(round(y_min_c)),
                    "x_max": int(round(x_max_c)),
                    "y_max": int(round(y_max_c)),
                    "width": int(round(width_px)),
                    "height": int(round(height_px)),
                },
                "box_normalized": {
                    "x_min": norm_xmin,
                    "y_min": norm_ymin,
                    "x_max": norm_xmax,
                    "y_max": norm_ymax,
                },
            }

            query_groups[matched_query].append(det_item)
            all_detections.append(det_item)

        # Sort all detections descending by confidence score and apply top_k cutoff
        all_detections.sort(key=lambda d: d["score"], reverse=True)
        if top_k and len(all_detections) > top_k:
            all_detections = all_detections[:top_k]

        # Structure query breakdown list
        queries_output: List[Dict[str, Any]] = []
        for q in valid_queries:
            group_dets = [d for d in all_detections if d["query"] == q]
            group_dets.sort(key=lambda d: d["score"], reverse=True)
            top_score = group_dets[0]["score"] if group_dets else 0.0
            queries_output.append(
                {
                    "query": q,
                    "prompt_applied": self.format_prompt(q, prompt_template),
                    "count": len(group_dets),
                    "top_score": top_score,
                    "detections": group_dets,
                }
            )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

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
            "mode": "open_vocabulary_detection",
            "image": image_info,
            "model": {
                "name": "OWL-ViT",
                "model_id": self.model_id,
                "parameters": self.param_count,
                "device": self.device,
            },
            "summary": {
                "total_queries": len(valid_queries),
                "total_detections": len(all_detections),
                "score_threshold": float(score_threshold),
                "prompt_template": prompt_template,
            },
            "queries": queries_output,
            "detections": all_detections,
            "timing": {
                "preprocessing_ms": round(prep_ms, 2),
                "inference_ms": round(inf_ms, 2),
                "total_ms": round(total_elapsed_ms, 2),
            },
        }
