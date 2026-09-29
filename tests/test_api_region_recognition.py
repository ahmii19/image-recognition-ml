"""Integration and contract tests for FastAPI Region-Level Open-Vocabulary Recognition Endpoints."""

import io
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.dependencies import get_open_vocab_engine, get_owlvit_detector
from src.api.main import app


@pytest.fixture
def mock_detector():
    """Create a mock OWLViTDetector returning deterministic detections."""
    detector = MagicMock()
    detector.validate_queries.side_effect = lambda q: [s.strip() for s in (q if isinstance(q, list) else q.split(",")) if s.strip()]
    detector.param_count = 153231879
    detector.detect_objects.return_value = {
        "success": True,
        "mode": "open_vocabulary_detection",
        "detections": [
            {
                "detection_id": "det-1",
                "query": "dog",
                "prompt_used": "a photo of a dog",
                "score": 0.88,
                "box": {"x_min": 20, "y_min": 30, "x_max": 200, "y_max": 220, "width": 180, "height": 190},
                "box_normalized": {"x_min": 0.06, "y_min": 0.12, "x_max": 0.62, "y_max": 0.91},
            },
            {
                "detection_id": "det-2",
                "query": "collar",
                "prompt_used": "a photo of a collar",
                "score": 0.62,
                "box": {"x_min": 80, "y_min": 90, "x_max": 140, "y_max": 120, "width": 60, "height": 30},
                "box_normalized": {"x_min": 0.25, "y_min": 0.37, "x_max": 0.43, "y_max": 0.50},
            },
        ],
    }
    return detector


@pytest.fixture
def mock_engine():
    """Create a mock OpenVocabEngine wrapping a mock classifier."""
    engine = MagicMock()
    classifier = MagicMock()
    classifier.model_name = "ViT-B-32"
    classifier.device = torch.device("cpu")
    classifier.preprocess = lambda img: torch.zeros((3, 224, 224), dtype=torch.float32)
    classifier.tokenizer = lambda texts: torch.zeros((len(texts), 77), dtype=torch.long)

    model = MagicMock()
    def mock_encode_text(tokens):
        num_q = tokens.shape[0]
        feats = torch.randn(num_q, 512)
        return feats / feats.norm(dim=-1, keepdim=True)

    def mock_encode_image(batch_tensor):
        batch_size = batch_tensor.shape[0]
        feats = torch.randn(batch_size, 512)
        return feats / feats.norm(dim=-1, keepdim=True)

    model.encode_text.side_effect = mock_encode_text
    model.encode_image.side_effect = mock_encode_image
    classifier.model = model
    engine._classifier = classifier
    return engine


@pytest.fixture
def client(mock_detector, mock_engine):
    """FastAPI TestClient with patched startup engines and dependency overrides."""
    app.dependency_overrides[get_owlvit_detector] = lambda: mock_detector
    app.dependency_overrides[get_open_vocab_engine] = lambda: mock_engine

    with patch("src.api.main.OpenVocabEngine", return_value=mock_engine), \
         patch("src.api.main.AdvancedRecognitionEngine", return_value=MagicMock()):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a sample PNG image encoded in memory for test uploads."""
    img = Image.new("RGB", (320, 240), color=(180, 120, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_models_endpoint_includes_region_intelligence(client):
    """Verify GET /api/v1/models advertises region_intelligence capability and mode."""
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()

    assert "region_intelligence" in data
    ri = data["region_intelligence"]
    assert ri["available"] is True
    assert ri["detector"] == "OWL-ViT base patch32"
    assert ri["recognizer"] == "OpenCLIP-ViT-B-32"
    assert ri["lazy_loaded"] is True
    assert ri["crop_padding_percent"] == 10
    assert "region_recognition" in data["supported_modes"]


def test_api_region_recognition_success(client, sample_image_bytes):
    """Verify POST /api/v1/open-vocabulary/region-recognition successfully executes and returns validated schema."""
    files = {"image": ("test_dog.png", sample_image_bytes, "image/png")}
    data = {
        "text_queries": "dog, collar, pavement",
        "top_k": "3",
        "detection_threshold": "0.01",
        "crop_padding_percent": "10",
    }

    res = client.post("/api/v1/open-vocabulary/region-recognition", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()

    assert resp_data["success"] is True
    assert resp_data["mode"] == "region_recognition"
    assert resp_data["summary"]["total_queries"] == 3
    assert resp_data["summary"]["crop_padding_percent"] == 10
    assert len(resp_data["regions"]) == 2
    assert resp_data["regions"][0]["region_id"] == "reg-01"
    assert resp_data["regions"][0]["detection"]["label"] == "dog"
    assert "timing" in resp_data
    assert resp_data["timing"]["total_pipeline_ms"] >= 0


def test_api_region_recognition_json_queries(client, sample_image_bytes):
    """Verify POST /api/v1/open-vocabulary/region-recognition supports JSON-encoded text query arrays."""
    files = {"image": ("test_dog.png", sample_image_bytes, "image/png")}
    data = {
        "text_queries": '["rose", "sunflower", "tulip"]',
        "top_k": "2",
        "detection_threshold": "0.01",
        "crop_padding_percent": "15",
    }

    res = client.post("/api/v1/open-vocabulary/region-recognition", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()

    assert resp_data["success"] is True
    assert resp_data["summary"]["total_queries"] == 3
    assert resp_data["summary"]["crop_padding_percent"] == 15


def test_api_region_recognition_empty_queries_rejection(client, sample_image_bytes):
    """Verify API rejects requests with empty text queries."""
    files = {"image": ("test_dog.png", sample_image_bytes, "image/png")}
    data = {"text_queries": "   "}

    res = client.post("/api/v1/open-vocabulary/region-recognition", files=files, data=data)
    assert res.status_code == 400
    error_data = res.json()
    assert error_data["success"] is False
    assert error_data["error"]["code"] == "INVALID_TEXT_QUERIES"


def test_api_region_recognition_too_many_queries(client, sample_image_bytes):
    """Verify API rejects requests exceeding MAX_TEXT_QUERIES limit (20)."""
    many_queries = ", ".join([f"query_{i}" for i in range(25)])
    files = {"image": ("test_dog.png", sample_image_bytes, "image/png")}
    data = {"text_queries": many_queries}

    res = client.post("/api/v1/open-vocabulary/region-recognition", files=files, data=data)
    assert res.status_code == 400
    error_data = res.json()
    assert error_data["success"] is False
    assert error_data["error"]["code"] == "TOO_MANY_QUERIES"
