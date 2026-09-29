"""Tests for Health, Readiness, and Model Metadata API Endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan context initialized."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /api/v1/health returns ok and service info."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "image-recognition-api"
    assert "version" in data
    assert data["models"]["classifier"] is True
    assert data["models"]["detector"] is True


def test_readiness_endpoint(client):
    """Test GET /api/v1/ready returns ready status and model availability."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["models_ready"] is True
    assert data["classifier_loaded"] is True
    assert data["detector_loaded"] is True


def test_models_metadata_endpoint(client):
    """Test GET /api/v1/models returns safe model metadata without filesystem paths."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()

    # Classifier info
    assert data["classifier"]["model"] == "MobileNetV2"
    assert data["classifier"]["vocabulary"] == "ImageNet-1K"
    assert data["classifier"]["classes"] == 1000

    # Detector info
    assert data["detector"]["model"] == "SSD-MobileNetV2-COCO"
    assert data["detector"]["vocabulary"] == "COCO-2017"
    assert data["detector"]["semantic_categories"] == 80

    # Formats & modes
    assert "jpg" in data["supported_formats"]
    assert "png" in data["supported_formats"]
    assert "all" in data["supported_modes"]
    assert "classification" in data["supported_modes"]
    assert "detection" in data["supported_modes"]
    assert data["max_upload_size_mb"] == 10


def test_openapi_schema_endpoint(client):
    """Test GET /openapi.json returns valid OpenAPI 3.x schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/api/v1/health" in data["paths"]
    assert "/api/v1/ready" in data["paths"]
    assert "/api/v1/models" in data["paths"]
    assert "/api/v1/recognize" in data["paths"]
    assert "/api/v1/classify" in data["paths"]
    assert "/api/v1/detect" in data["paths"]


def test_docs_redirect(client):
    """Test root redirect to /docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308, 302, 301)
    assert response.headers["location"] == "/docs"
