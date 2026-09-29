"""Integration Tests for FastAPI Recognition, Classification, and Detection Endpoints."""

import concurrent.futures
import io
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan context initialized."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_image_bytes():
    """Create sample JPEG image bytes for tests."""
    existing = Path("data/phase2/dataset_cache/flower_photos/flower_photos/roses/10090824183_d02c613f10_m.jpg")
    if existing.exists():
        return existing.read_bytes()

    img = Image.new("RGB", (300, 300), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_recognize_all_mode(client, sample_image_bytes):
    """Test POST /api/v1/recognize with mode=all executes classification and detection."""
    files = {"image": ("test_rose.jpg", sample_image_bytes, "image/jpeg")}
    data = {"mode": "all", "classification_threshold": "0.3", "detection_threshold": "0.3"}

    response = client.post("/api/v1/recognize", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["mode"] == "all"
    assert res["image"]["file_name"] == "test_rose.jpg"
    assert res["image"]["channels"] == 3
    assert res["metadata"]["engine"] == "AdvancedRecognitionEngine-v3"
    assert res["metadata"]["classifier_model"] is not None
    assert res["metadata"]["detector_model"] is not None
    assert "classification" in res
    assert res["classification"]["success"] is True
    assert len(res["classification"]["top5"]) > 0
    assert "top1" in res["classification"]
    assert "detection" in res
    assert res["detection"]["success"] is True
    assert "objects" in res["detection"]
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_recognize_classification_mode(client, sample_image_bytes):
    """Test POST /api/v1/recognize with mode=classification."""
    files = {"image": ("test_img.jpg", sample_image_bytes, "image/jpeg")}
    data = {"mode": "classification", "top_k": "3"}

    response = client.post("/api/v1/recognize", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["mode"] == "classification"
    assert res["classification"] is not None
    assert len(res["classification"]["predictions"]) == 3
    assert res["detection"] is None


def test_recognize_detection_mode(client, sample_image_bytes):
    """Test POST /api/v1/recognize with mode=detection."""
    files = {"image": ("test_img.jpg", sample_image_bytes, "image/jpeg")}
    data = {"mode": "detection", "detection_threshold": "0.3"}

    response = client.post("/api/v1/recognize", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["mode"] == "detection"
    assert res["classification"] is None
    assert res["detection"] is not None
    assert res["detection"]["model"] == "SSD-MobileNetV2-COCO"


def test_standalone_classify_endpoint(client, sample_image_bytes):
    """Test POST /api/v1/classify returns classification result directly."""
    files = {"image": ("flower.jpg", sample_image_bytes, "image/jpeg")}
    data = {"top_k": "5", "threshold": "0.2"}

    response = client.post("/api/v1/classify", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert "MobileNetV2" in res["model"]
    assert len(res["predictions"]) == 5
    assert res["top1"] is not None
    assert 0.0 <= res["top1"]["confidence"] <= 1.0


def test_standalone_detect_endpoint(client, sample_image_bytes):
    """Test POST /api/v1/detect returns detection result directly."""
    files = {"image": ("objects.jpg", sample_image_bytes, "image/jpeg")}
    data = {"threshold": "0.2", "max_detections": "10"}

    response = client.post("/api/v1/detect", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["model"] == "SSD-MobileNetV2-COCO"
    assert isinstance(res["objects"], list)
    assert "inference_ms" in res


def test_concurrent_api_requests(client, sample_image_bytes):
    """Test multiple concurrent requests to verify thread safety and lack of model reloading."""
    def send_request(idx):
        files = {"image": (f"test_{idx}.jpg", sample_image_bytes, "image/jpeg")}
        data = {"mode": "classification", "top_k": "2"}
        resp = client.post("/api/v1/recognize", files=files, data=data)
        return resp.status_code, resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_request, i) for i in range(4)]
        results = [f.result() for f in futures]

    for status_code, data in results:
        assert status_code == 200
        assert data["success"] is True
        assert data["classification"] is not None
