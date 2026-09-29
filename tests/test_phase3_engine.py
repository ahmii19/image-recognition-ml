"""Unit and integration tests for Phase 3 Classifier, Detector, Visualizer, and Unified Engine."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.phase3.classifier import Phase3Classifier
from src.phase3.engine import AdvancedRecognitionEngine
from src.phase3.visualizer import render_detections


@pytest.fixture(scope="module")
def classifier_instance():
    """Reusable classifier instance across tests."""
    return Phase3Classifier()


@pytest.fixture(scope="module")
def recognition_engine():
    """Reusable unified recognition engine instance across tests."""
    return AdvancedRecognitionEngine(lazy_load=False)


def test_classifier_output_schema_and_top_k(classifier_instance):
    """Verify ImageNet classifier returns valid top-1, top-5, and schema fields."""
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    res = classifier_instance.classify(img, top_k=5)

    assert res["success"] is True
    assert res["mode"] == "classification"
    assert "top1" in res
    assert "top5" in res
    assert len(res["top5"]) == 5
    assert "label" in res["top1"]
    assert "confidence" in res["top1"]
    assert 0.0 <= res["top1"]["confidence"] <= 1.0
    assert 0.0 <= res["top1"]["confidence_percent"] <= 100.0
    assert "inference_ms" in res
    assert "model" in res


def test_classifier_confidence_threshold_behavior(classifier_instance):
    """Verify classification status updates based on confidence threshold."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res_high_thresh = classifier_instance.classify(img, threshold=0.99)
    # A uniform gray image should have low confidence (<99%)
    assert res_high_thresh["is_confident"] is False
    assert res_high_thresh["status"] == "low_confidence"


def test_visualization_generation():
    """Verify bounding box rendering generates valid image file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        mock_detections = [
            {
                "label": "person",
                "confidence": 0.95,
                "confidence_percent": 95.0,
                "box": {"x1": 50, "y1": 50, "x2": 200, "y2": 350},
            },
            {
                "label": "dog",
                "confidence": 0.88,
                "confidence_percent": 88.0,
                "box": {"x1": 220, "y1": 150, "x2": 380, "y2": 350},
            },
        ]
        out_path = Path(tmpdir) / "rendered.png"
        saved = render_detections(img, mock_detections, save_path=out_path)

        assert saved.exists()
        assert saved.stat().st_size > 0
        with Image.open(saved) as loaded:
            assert loaded.size == (400, 400)


def test_unified_engine_classification_mode(recognition_engine):
    """Verify unified engine execution in classification-only mode."""
    img = Image.new("RGB", (300, 300), color=(100, 200, 150))
    res = recognition_engine.recognize(img, mode="classification")

    assert res["success"] is True
    assert res["mode"] == "classification"
    assert "classification" in res
    assert "detection" not in res
    assert res["image"]["width"] == 300
    assert res["image"]["height"] == 300


def test_unified_engine_detection_mode(recognition_engine):
    """Verify unified engine execution in detection-only mode."""
    img = Image.new("RGB", (320, 320), color=(50, 50, 50))
    res = recognition_engine.recognize(img, mode="detection")

    assert res["success"] is True
    assert res["mode"] == "detection"
    assert "detection" in res
    assert "classification" not in res
    assert "objects" in res["detection"]
    assert "count" in res["detection"]


def test_unified_engine_all_mode_and_bounding_boxes(recognition_engine):
    """Verify unified engine runs both classification and detection with valid boxes."""
    img = Image.new("RGB", (500, 400), color=(80, 120, 160))
    res = recognition_engine.recognize(img, mode="all", render_visualization=True)

    assert res["success"] is True
    assert res["mode"] == "all"
    assert "classification" in res
    assert "detection" in res
    assert "total_inference_ms" in res["metadata"]

    # Verify bounding box geometry invariants for any detected objects
    for obj in res["detection"]["objects"]:
        box = obj["box"]
        assert 0 <= box["x1"] < box["x2"] <= 500
        assert 0 <= box["y1"] < box["y2"] <= 400
        assert box["width"] > 0
        assert box["height"] > 0
        assert 0.0 <= obj["confidence"] <= 1.0


def test_unified_engine_invalid_input_handling(recognition_engine):
    """Verify unified engine gracefully returns structured error for bad inputs."""
    res = recognition_engine.recognize("invalid_missing_file.jpg", mode="all")
    assert res["success"] is False
    assert "error" in res
    assert res["error_code"] == "FILE_NOT_FOUND"
