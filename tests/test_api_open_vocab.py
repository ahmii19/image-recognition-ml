"""Integration tests for FastAPI Open-Vocabulary API Route."""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create test client with full lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_rose_bytes():
    """Load sample rose image bytes."""
    rose_path = Path("data/phase2/dataset_cache/flower_photos/flower_photos/roses/10090824183_d02c613f10_m.jpg")
    if rose_path.exists():
        return rose_path.read_bytes()

    img = Image.new("RGB", (300, 300), color=(200, 40, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_open_vocab_endpoint_success(client, sample_rose_bytes):
    """Test POST /api/v1/open-vocabulary with comma-separated text_queries."""
    files = {"image": ("rose_test.jpg", sample_rose_bytes, "image/jpeg")}
    data = {
        "text_queries": "rose, sunflower, sports car, laptop, airplane",
        "top_k": "5",
    }

    response = client.post("/api/v1/open-vocabulary", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert "open_vocabulary" in res
    assert "metadata" in res
    assert res["open_vocabulary"]["top_match"]["query"] == "rose"
    assert len(res["open_vocabulary"]["results"]) == 5
    assert res["image"]["file_name"] == "rose_test.jpg"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_open_vocab_endpoint_json_queries(client, sample_rose_bytes):
    """Test POST /api/v1/open-vocabulary with JSON array text_queries."""
    files = {"image": ("rose_test.jpg", sample_rose_bytes, "image/jpeg")}
    data = {
        "text_queries": '["rose", "yellow sunflower", "red sports car"]',
        "top_k": "3",
    }

    response = client.post("/api/v1/open-vocabulary", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["open_vocabulary"]["top_match"]["query"] == "rose"
    assert len(res["open_vocabulary"]["results"]) == 3


def test_open_vocab_endpoint_custom_prompt_template(client, sample_rose_bytes):
    """Test POST /api/v1/open-vocabulary with custom prompt template."""
    files = {"image": ("rose_test.jpg", sample_rose_bytes, "image/jpeg")}
    data = {
        "text_queries": "rose, daisy",
        "prompt_template": "a close-up photo of a blooming {}",
    }

    response = client.post("/api/v1/open-vocabulary", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["open_vocabulary"]["prompt_template"] == "a close-up photo of a blooming {}"



def test_open_vocab_endpoint_empty_queries_rejection(client, sample_rose_bytes):
    """Test POST /api/v1/open-vocabulary rejects empty query string."""
    files = {"image": ("rose_test.jpg", sample_rose_bytes, "image/jpeg")}
    data = {"text_queries": "   ,  "}

    response = client.post("/api/v1/open-vocabulary", files=files, data=data)
    assert response.status_code in (400, 422)
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_TEXT_QUERIES"


def test_open_vocab_endpoint_too_many_queries(client, sample_rose_bytes):
    """Test POST /api/v1/open-vocabulary rejects > 20 queries."""
    files = {"image": ("rose_test.jpg", sample_rose_bytes, "image/jpeg")}
    queries = [f"concept_{i}" for i in range(25)]
    data = {"text_queries": ", ".join(queries)}

    response = client.post("/api/v1/open-vocabulary", files=files, data=data)
    assert response.status_code in (400, 422)
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "TOO_MANY_QUERIES"


def test_models_metadata_includes_open_vocab(client):
    """Test GET /api/v1/models includes open_vocabulary architecture information."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    res = response.json()

    assert "open_vocabulary" in res
    assert res["open_vocabulary"] is not None
    assert res["open_vocabulary"]["model"] == "OpenCLIP-ViT-B-32"
    assert res["open_vocabulary"]["pretrained"] == "laion2b_s34b_b79k"
    assert res["open_vocabulary"]["supports_text_queries"] is True
    assert res["open_vocabulary"]["supports_bounding_boxes"] is False
    assert res["open_vocabulary"]["max_queries"] == 20
    assert "open_vocabulary" in res["supported_modes"]
