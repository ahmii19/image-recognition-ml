"""Integration tests for OWL-ViT Open-Vocabulary Object Detection API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.phase6.owlvit.lifecycle import OWLViTLifecycleManager


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean state before and after tests."""
    yield
    OWLViTLifecycleManager.unload()


@pytest.fixture
def client():
    """FastAPI TestClient with app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate in-memory JPEG bytes for multipart upload."""
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_models_endpoint_reports_owlvit(client):
    """Verify /models endpoint lists open_vocabulary_detection capability."""
    OWLViTLifecycleManager.unload()
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    
    assert "open_vocabulary_detection" in data
    ov_det = data["open_vocabulary_detection"]
    assert ov_det["model"] == "OWL-ViT base patch32"
    assert ov_det["supports_bounding_boxes"] is True
    assert ov_det["supports_text_queries"] is True
    assert ov_det["lazy_loaded"] is True
    assert ov_det["loaded"] is False


def test_api_detect_open_vocabulary_success(client, sample_image_bytes):
    """Verify POST /open-vocabulary/detect returns valid detections."""
    response = client.post(
        "/api/v1/open-vocabulary/detect",
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={
            "text_queries": "blue square, pattern, object",
            "top_k": "10",
            "score_threshold": "0.01",
            "prompt_template": "a photo of a {}",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["mode"] == "open_vocabulary_detection"
    assert data["image"]["width"] == 300
    assert data["image"]["height"] == 300
    assert len(data["queries"]) == 3
    assert "detections" in data
    assert "timing" in data
    assert data["timing"]["total_ms"] > 0

    # Verify models endpoint now reports loaded=True
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    assert models_res.json()["open_vocabulary_detection"]["loaded"] is True


def test_api_detect_empty_queries_rejection(client, sample_image_bytes):
    """Verify 422/validation rejection when empty query is provided."""
    response = client.post(
        "/api/v1/open-vocabulary/detect",
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"text_queries": "   ,   "},
    )
    assert response.status_code in (400, 422)
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_TEXT_QUERIES"


def test_api_detect_too_many_queries(client, sample_image_bytes):
    """Verify rejection when queries exceed MAX_TEXT_QUERIES."""
    too_many = ",".join([f"q{i}" for i in range(25)])
    response = client.post(
        "/api/v1/open-vocabulary/detect",
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"text_queries": too_many},
    )
    assert response.status_code in (400, 422)
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "TOO_MANY_QUERIES"


def test_api_unload_and_status(client, sample_image_bytes):
    """Verify /open-vocabulary/unload and /open-vocabulary/status endpoints."""
    # 1. Trigger detection to load model
    client.post(
        "/api/v1/open-vocabulary/detect",
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"text_queries": "square"},
    )
    assert OWLViTLifecycleManager.is_loaded() is True

    # 2. Check status endpoint
    status_res = client.get("/api/v1/open-vocabulary/status")
    assert status_res.status_code == 200
    assert status_res.json()["loaded"] is True

    # 3. Call unload
    unload_res = client.post("/api/v1/open-vocabulary/unload")
    assert unload_res.status_code == 200
    assert unload_res.json()["unloaded"] is True
    assert OWLViTLifecycleManager.is_loaded() is False
