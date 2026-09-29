"""OpenCLIP ViT-B/32 Zero-Shot Vision-Language Classifier for Phase 6A."""

import logging
import time
from typing import Any, Dict, List, Optional, Union
import open_clip
from PIL import Image
import torch
import torch.nn.functional as F

from src.device import resolve_device
from src.phase6.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOP_K,
    OPENCLIP_DEVICE,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_NUM_THREADS,
    OPENCLIP_PRETRAINED,
    ensure_phase6_directories,
)

logger = logging.getLogger("phase6.open_vocab")


class OpenVocabClassifier:
    """Zero-Shot Vision-Language Classifier using OpenCLIP supporting CPU and CUDA."""

    def __init__(
        self,
        model_name: str = OPENCLIP_MODEL_NAME,
        pretrained: str = OPENCLIP_PRETRAINED,
        num_threads: int = OPENCLIP_NUM_THREADS,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize OpenCLIP backbone, preprocessing pipeline, and tokenizer.

        Args:
            model_name: Architecture tag (e.g. 'ViT-B-32').
            pretrained: Checkpoint tag (e.g. 'laion2b_s34b_b79k' or 'openai').
            num_threads: Number of CPU worker threads for PyTorch.
            device: Device target ('cpu', 'cuda', or 'auto' / None).
        """
        ensure_phase6_directories()
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_threads = num_threads
        self.device = resolve_device(device if device is not None else OPENCLIP_DEVICE)

        # Configure CPU threading if running on CPU to respect host cores
        if self.device.type == "cpu" and self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        logger.info(
            f"Initializing OpenCLIP Classifier ({self.model_name} / {self.pretrained}) on {self.device}..."
        )
        init_start = time.perf_counter()

        # Load model and preprocessing transform
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Warmup forward pass on target device
        with torch.no_grad():
            dummy_img = torch.zeros((1, 3, 224, 224), device=self.device)
            dummy_text = self.tokenizer(["a photo of an object"]).to(self.device)
            _ = self.model.encode_image(dummy_img)
            _ = self.model.encode_text(dummy_text)

        init_ms = (time.perf_counter() - init_start) * 1000.0
        logger.info(
            f"OpenCLIP Classifier ({self.model_name}) ready on {self.device} in {init_ms:.1f}ms."
        )

    def classify_zero_shot(
        self,
        pil_image: Image.Image,
        text_queries: List[str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Perform zero-shot similarity matching between image and text queries.

        Args:
            pil_image: Validated 3-channel RGB PIL Image.
            text_queries: List of candidate concept strings.
            prompt_template: Template string containing '{}' for query interpolation.
            min_similarity: Minimum cosine similarity threshold for positive match heuristic.
            top_k: Maximum number of ranked results to return.

        Returns:
            Structured dictionary with similarity scores, ranking, and telemetry.
        """
        total_start = time.perf_counter()

        # Step 1: Preprocess Image
        prep_start = time.perf_counter()
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # Step 2: Format & Tokenize Text Queries
        # Apply prompt template (e.g. "a photo of a {}")
        formatted_prompts = [
            prompt_template.format(q.strip()) if "{}" in prompt_template else f"{prompt_template} {q.strip()}"
            for q in text_queries
        ]
        text_tokens = self.tokenizer(formatted_prompts).to(self.device)

        # Step 3: Compute Vision and Text Embeddings
        inf_start = time.perf_counter()
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            # Step 4: L2-Normalize Embeddings to Unit Hypersphere
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Step 5: Compute Cosine Similarity Dot Product
            # (1, D) @ (N, D).T -> (1, N)
            similarity_matrix = (image_features @ text_features.T).squeeze(0)
            raw_scores = similarity_matrix.cpu().numpy().tolist()

        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        # Step 6: Rank Results by Similarity in Descending Order
        paired = [
            {"query": text_queries[i], "similarity_score": round(float(raw_scores[i]), 4)}
            for i in range(len(text_queries))
        ]
        paired.sort(key=lambda x: x["similarity_score"], reverse=True)

        ranked_results: List[Dict[str, Any]] = []
        for rank, item in enumerate(paired[:top_k], start=1):
            ranked_results.append(
                {
                    "query": item["query"],
                    "similarity_score": item["similarity_score"],
                    "rank": rank,
                }
            )

        top_match = ranked_results[0] if ranked_results else None
        no_match = bool(top_match is None or top_match["similarity_score"] < min_similarity)

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "success": True,
            "mode": "open_vocabulary",
            "open_vocabulary": {
                "model": f"OpenCLIP-{self.model_name}",
                "pretrained": self.pretrained,
                "prompt_template": prompt_template,
                "queries": text_queries,
                "results": ranked_results,
                "top_match": top_match,
                "no_match": no_match,
                "min_similarity_threshold": min_similarity,
                "inference_ms": round(inf_ms, 2),
                "preprocessing_ms": round(prep_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    def classify(
        self,
        pil_image: Image.Image,
        queries: Optional[List[str]] = None,
        text_queries: Optional[List[str]] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        similarity_threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Convenience alias for classify_zero_shot supporting flexible query keywords."""
        q_list = queries if queries is not None else text_queries
        if q_list is None or len(q_list) == 0:
            raise ValueError("At least one non-empty text query must be provided.")

        cutoff = similarity_threshold if similarity_threshold is not None else min_similarity
        raw_res = self.classify_zero_shot(
            pil_image=pil_image,
            text_queries=q_list,
            prompt_template=prompt_template,
            min_similarity=cutoff,
            top_k=top_k,
        )
        ov = raw_res["open_vocabulary"]
        filtered_ranked = [
            item for item in ov["results"]
            if (similarity_threshold is None or item["similarity_score"] >= similarity_threshold)
        ]
        return {
            "success": True,
            "top_match": ov["top_match"],
            "all_ranked": filtered_ranked,
            "total_queries_evaluated": len(q_list),
            "inference_time_ms": ov["inference_ms"],
            "prompt_template": prompt_template,
            "open_vocabulary": ov,
        }


# Model alias for backward/forward naming compatibility
OpenCLIPClassifier = OpenVocabClassifier

