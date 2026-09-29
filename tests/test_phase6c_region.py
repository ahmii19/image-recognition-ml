"""Unit and validation tests for Phase 6C Region-Level Recognition Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import pytest
import torch

from src.phase6.config import (
    DEFAULT_CROP_PADDING_PERCENT,
    DEFAULT_MAX_REGIONS,
    DEFAULT_PROMPT_TEMPLATE,
    OPENCLIP_MODEL_NAME,
)
from src.phase6.region_pipeline import (
    RegionRecognitionPipeline,
    compute_padded_crop_box,
)


@pytest.fixture
def mock_detector():
    """Create a lightweight mock OWLViTDetector."""
    detector = MagicMock()
    detector.validate_queries.side_effect = lambda q: [s.strip() for s in (q if isinstance(q, list) else q.split(",")) if s.strip()]
    detector.param_count = 153231879

    # Return realistic mock detections
    detector.detect_objects.return_value = {
        "success": True,
        "mode": "open_vocabulary_detection",
        "detections": [
            {
                "detection_id": "det-1",
                "query": "dog",
                "prompt_used": "a photo of a dog",
                "score": 0.85,
                "box": {"x_min": 50, "y_min": 60, "x_max": 250, "y_max": 260, "width": 200, "height": 200},
                "box_normalized": {"x_min": 0.1, "y_min": 0.12, "x_max": 0.5, "y_max": 0.52},
            },
            {
                "detection_id": "det-2",
                "query": "collar",
                "prompt_used": "a photo of a collar",
                "score": 0.65,
                "box": {"x_min": 100, "y_min": 120, "x_max": 180, "y_max": 160, "width": 80, "height": 40},
                "box_normalized": {"x_min": 0.2, "y_min": 0.24, "x_max": 0.36, "y_max": 0.32},
            },
            {
                "detection_id": "det-3",
                "query": "pavement",
                "prompt_used": "a photo of a pavement",
                "score": 0.45,
                "box": {"x_min": 0, "y_min": 200, "x_max": 500, "y_max": 500, "width": 500, "height": 300},
                "box_normalized": {"x_min": 0.0, "y_min": 0.4, "x_max": 1.0, "y_max": 1.0},
            },
        ],
    }
    return detector


@pytest.fixture
def mock_classifier():
    """Create a lightweight mock OpenCLIPClassifier with simulated embedding projections."""
    classifier = MagicMock()
    classifier.model_name = "ViT-B-32"
    classifier.device = torch.device("cpu")
    classifier.preprocess = lambda img: torch.zeros((3, 224, 224), dtype=torch.float32)
    classifier.tokenizer = lambda texts: torch.zeros((len(texts), 77), dtype=torch.long)

    model = MagicMock()
    # Mock text encoding: returns normalized random features (Q, 512)
    def mock_encode_text(tokens):
        num_q = tokens.shape[0]
        feats = torch.randn(num_q, 512)
        return feats / feats.norm(dim=-1, keepdim=True)

    # Mock image encoding: returns normalized random features (B, 512)
    def mock_encode_image(batch_tensor):
        batch_size = batch_tensor.shape[0]
        feats = torch.randn(batch_size, 512)
        return feats / feats.norm(dim=-1, keepdim=True)

    model.encode_text.side_effect = mock_encode_text
    model.encode_image.side_effect = mock_encode_image
    classifier.model = model
    return classifier


@pytest.fixture
def sample_test_image():
    """Create a clean 500x500 RGB test image."""
    return Image.new("RGB", (500, 500), color=(180, 140, 100))


def test_compute_padded_crop_box_exact_10_percent():
    """Verify 10% symmetrical context padding calculation expands w and h by exactly 10% on each side."""
    box = {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}  # 100x100 box
    # 10% of 100 = 10px pad on all sides -> [90, 90, 210, 210]
    x1, y1, x2, y2 = compute_padded_crop_box(box, image_width=500, image_height=500, padding_percent=10)
    assert x1 == 90
    assert y1 == 90
    assert x2 == 210
    assert y2 == 210
    assert (x2 - x1) == 120
    assert (y2 - y1) == 120


def test_compute_padded_crop_box_boundary_clamping():
    """Verify context padding is strictly clamped to image borders without out-of-bound coordinates."""
    box = {"x_min": 5, "y_min": 5, "x_max": 95, "y_max": 95}  # Near top-left origin
    x1, y1, x2, y2 = compute_padded_crop_box(box, image_width=100, image_height=100, padding_percent=20)
    assert x1 == 0  # Clamped to 0
    assert y1 == 0  # Clamped to 0
    assert x2 == 100  # Clamped to image_width
    assert y2 == 100  # Clamped to image_height


def test_compute_padded_crop_box_zero_or_degenerate_handling():
    """Verify degenerate boxes produce a valid safe minimal area."""
    box = {"x_min": 50, "y_min": 50, "x_max": 51, "y_max": 51}
    x1, y1, x2, y2 = compute_padded_crop_box(box, image_width=200, image_height=200, padding_percent=10)
    assert x2 > x1
    assert y2 > y1
    assert (x2 - x1) >= 4
    assert (y2 - y1) >= 4


def test_region_selection_and_limit_enforcement(mock_detector, mock_classifier, sample_test_image):
    """Verify max_regions caps the number of candidate regions deterministically by score/area."""
    pipeline = RegionRecognitionPipeline(detector=mock_detector, classifier=mock_classifier)
    queries = ["dog", "collar", "pavement"]
    res = pipeline.process_image(
        image_input=sample_test_image,
        text_queries=queries,
        detection_threshold=0.05,
        max_regions=2,  # Strict cap of 2
        crop_padding_percent=10,
    )

    assert res["success"] is True
    assert res["mode"] == "region_recognition"
    assert len(res["regions"]) == 2
    assert res["summary"]["analyzed_regions_count"] == 2
    # Highest scoring region (score 0.85) should be reg-01
    assert res["regions"][0]["detection"]["score"] == 0.85


def test_text_embedding_reuse_single_text_encode_call(mock_detector, mock_classifier, sample_test_image):
    """Explicitly prove that text embeddings are encoded once and reused across all N regions."""
    pipeline = RegionRecognitionPipeline(detector=mock_detector, classifier=mock_classifier)
    queries = ["dog", "cat", "car", "person"]

    res = pipeline.process_image(
        image_input=sample_test_image,
        text_queries=queries,
        detection_threshold=0.08,
        max_regions=5,
    )

    assert res["success"] is True
    # encode_text MUST be called exactly ONCE for the whole request regardless of region count
    assert mock_classifier.model.encode_text.call_count == 1


def test_batched_image_encoding_single_batch_pass(mock_detector, mock_classifier, sample_test_image):
    """Explicitly prove that multiple region crops are stacked into a single batch tensor for encode_image."""
    pipeline = RegionRecognitionPipeline(
        detector=mock_detector,
        classifier=mock_classifier,
        batch_size=20,
    )

    res = pipeline.process_image(
        image_input=sample_test_image,
        text_queries=["dog", "collar", "pavement"],
        detection_threshold=0.08,
        max_regions=10,
    )

    assert res["success"] is True
    num_regions = len(res["regions"])
    assert num_regions == 3
    # Exactly 1 batch encode_image call containing all 3 crops
    assert mock_classifier.model.encode_image.call_count == 1


def test_end_to_end_region_pipeline_response_schema(mock_detector, mock_classifier, sample_test_image):
    """Verify complete output schema conforms to Phase 6C contract."""
    pipeline = RegionRecognitionPipeline(detector=mock_detector, classifier=mock_classifier)
    res = pipeline.process_image(
        image_input=sample_test_image,
        text_queries=["dog", "cat", "car"],
        detection_threshold=0.10,
        crop_padding_percent=10,
        top_k=3,
    )

    assert res["success"] is True
    assert res["mode"] == "region_recognition"
    assert "image" in res
    assert "summary" in res
    assert "models" in res
    assert "regions" in res
    assert "timing" in res

    assert res["summary"]["crop_padding_percent"] == 10
    assert res["timing"]["owlvit_detection_ms"] >= 0
    assert res["timing"]["openclip_batch_ms"] >= 0
    assert res["timing"]["total_pipeline_ms"] >= 0

    for reg in res["regions"]:
        assert "region_id" in reg
        assert "detection" in reg
        assert "original_box" in reg
        assert "padded_box" in reg
        assert "normalized_box" in reg
        assert "crop" in reg
        assert "recognition" in reg
        assert reg["crop"]["padding_percent"] == 10
        assert 0.0 <= reg["normalized_box"]["x_min"] <= 1.0
        assert 0.0 <= reg["normalized_box"]["y_min"] <= 1.0
        assert 0.0 <= reg["normalized_box"]["x_max"] <= 1.0
        assert 0.0 <= reg["normalized_box"]["y_max"] <= 1.0
        if reg["recognition"]:
            assert reg["recognition"]["top_match"] is not None
            assert -1.0 <= reg["recognition"]["top_match"]["similarity_score"] <= 1.0
