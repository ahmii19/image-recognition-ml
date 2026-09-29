"""Unit tests for Phase 6B OWL-ViT Open-Vocabulary Object Detection."""

import io
import pytest
from PIL import Image
import torch

from src.phase6.owlvit.config import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SCORE_THRESHOLD,
    MAX_TEXT_QUERIES,
    MAX_TEXT_QUERY_LENGTH,
    OWL_VIT_MODEL_ID,
)
from src.phase6.owlvit.detector import OWLViTDetector
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager


@pytest.fixture(autouse=True)
def cleanup_owlvit():
    """Ensure clean singleton state before and after each test."""
    yield
    OWLViTLifecycleManager.unload()


@pytest.fixture
def sample_image() -> Image.Image:
    """Create synthetic test image (200x200 RGB)."""
    return Image.new("RGB", (200, 200), color=(120, 180, 240))


def test_owlvit_config_defaults():
    """Verify configuration parameters and boundary constants."""
    assert OWL_VIT_MODEL_ID == "google/owlvit-base-patch32"
    assert MAX_TEXT_QUERIES == 20
    assert MAX_TEXT_QUERY_LENGTH == 128
    assert "{}" in DEFAULT_PROMPT_TEMPLATE
    assert 0.0 <= DEFAULT_SCORE_THRESHOLD <= 1.0


def test_query_validation_sanitization():
    """Verify query string normalization, unicode handling, and deduplication."""
    # Test on a dummy or instantiated detector
    detector = OWLViTDetector.__new__(OWLViTDetector)
    
    # Comma-separated string with spaces and duplicates
    queries = detector.validate_queries("dog,  CAT , dog, bird ")
    assert queries == ["dog", "CAT", "bird"]

    # List of queries with control characters
    queries = detector.validate_queries(["\x00rose\x1f", "sunflower\n"])
    assert queries == ["rose", "sunflower"]


def test_query_validation_errors():
    """Verify bounds enforcement for empty, too long, or too many queries."""
    detector = OWLViTDetector.__new__(OWLViTDetector)

    # Empty queries
    with pytest.raises(ValueError, match="At least 1 candidate text query"):
        detector.validate_queries([])

    with pytest.raises(ValueError, match="At least 1 candidate text query"):
        detector.validate_queries("  ,   ")

    # Too many queries (>20)
    with pytest.raises(ValueError, match="Maximum of 20 queries allowed"):
        detector.validate_queries([f"query_{i}" for i in range(25)])

    # Query too long (>128 chars)
    with pytest.raises(ValueError, match="exceeds maximum length"):
        detector.validate_queries(["a" * 150])


def test_prompt_formatting():
    """Verify prompt formatting and fallback on invalid templates."""
    detector = OWLViTDetector.__new__(OWLViTDetector)

    # Standard template
    formatted = detector.format_prompt("dog", "a photo of a {}")
    assert formatted == "a photo of a dog"

    # Custom valid template
    formatted = detector.format_prompt("red car", "an image showing {} in high resolution")
    assert formatted == "an image showing red car in high resolution"

    # Invalid template missing {} -> safe fallback
    formatted = detector.format_prompt("dog", "invalid template without placeholder")
    assert formatted == "a photo of a dog"


def test_lifecycle_lazy_loading_and_singleton():
    """Verify OWL-ViT starts unloaded and loads as a thread-safe singleton."""
    OWLViTLifecycleManager.unload()
    assert not OWLViTLifecycleManager.is_loaded()

    # First access -> loads model
    detector1 = OWLViTLifecycleManager.get_detector()
    assert detector1 is not None
    assert OWLViTLifecycleManager.is_loaded()

    # Second access -> returns identical instance
    detector2 = OWLViTLifecycleManager.get_detector()
    assert detector1 is detector2

    status = OWLViTLifecycleManager.get_status()
    assert status["loaded"] is True
    assert status["load_count"] >= 1
    assert status["total_inferences"] >= 2


def test_lifecycle_unload():
    """Verify explicit memory release."""
    detector = OWLViTLifecycleManager.get_detector()
    assert OWLViTLifecycleManager.is_loaded()

    unloaded = OWLViTLifecycleManager.unload()
    assert unloaded is True
    assert not OWLViTLifecycleManager.is_loaded()

    # Second unload returns False
    unloaded_again = OWLViTLifecycleManager.unload()
    assert unloaded_again is False


def test_real_owlvit_detection(sample_image):
    """Verify end-to-end detection inference and output schema adherence."""
    detector = OWLViTLifecycleManager.get_detector()
    
    result = detector.detect_objects(
        image_input=sample_image,
        text_queries=["square", "background", "pattern"],
        score_threshold=0.01,
        top_k=10,
        prompt_template="a photo of a {}",
    )

    assert result["success"] is True
    assert result["mode"] == "open_vocabulary_detection"
    assert "image" in result
    assert result["image"]["width"] == 200
    assert result["image"]["height"] == 200
    assert "model" in result
    assert result["model"]["name"] == "OWL-ViT"
    assert "queries" in result
    assert len(result["queries"]) == 3
    assert "detections" in result
    assert "timing" in result
    assert result["timing"]["inference_ms"] > 0

    for det in result["detections"]:
        assert "detection_id" in det
        assert det["detection_id"].startswith("det-")
        assert "box" in det
        assert "box_normalized" in det
        assert 0.0 <= det["box_normalized"]["x_min"] <= 1.0
        assert 0.0 <= det["box_normalized"]["y_min"] <= 1.0
        assert 0.0 <= det["box_normalized"]["x_max"] <= 1.0
        assert 0.0 <= det["box_normalized"]["y_max"] <= 1.0
