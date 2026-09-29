"""Tests for API Validation, Upload Constraints, Error Handling, and Hardening."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan context initialized."""
    with TestClient(app) as test_client:
        yield test_client


def test_missing_image_file_returns_422(client):
    """Test POST /api/v1/recognize without image multipart field returns 422."""
    response = client.post("/api/v1/recognize", data={"mode": "all"})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "INVALID_REQUEST_PARAMETERS"


def test_empty_zero_byte_file_returns_400(client):
    """Test POST /api/v1/recognize with 0-byte file returns 400 and EMPTY_FILE code."""
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/v1/recognize", files=files, data={"mode": "all"})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "EMPTY_FILE"


def test_corrupted_image_file_returns_400(client):
    """Test POST /api/v1/recognize with random invalid bytes returns 400."""
    files = {"image": ("corrupted.jpg", b"NOT_A_VALID_IMAGE_BYTES_12345", "image/jpeg")}
    response = client.post("/api/v1/recognize", files=files, data={"mode": "all"})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "CORRUPTED_IMAGE"


def test_unsupported_file_extension_returns_400(client):
    """Test POST /api/v1/recognize with .exe or .txt extension returns 400."""
    files = {"image": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")}
    response = client.post("/api/v1/recognize", files=files, data={"mode": "all"})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_invalid_mode_returns_422(client):
    """Test POST /api/v1/recognize with invalid mode returns 422 and INVALID_MODE code."""
    # Create valid minimal 1x1 GIF or PNG
    valid_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10"
        b"\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x1bIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"image": ("test.png", valid_png, "image/png")}
    response = client.post("/api/v1/recognize", files=files, data={"mode": "telepathy"})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_MODE"


def test_oversized_upload_returns_413(client, monkeypatch):
    """Test uploading a file larger than configured limit returns 413."""
    # Mock max upload limit to 1KB for fast testing
    from src.api import dependencies
    monkeypatch.setattr(dependencies, "MAX_UPLOAD_SIZE_BYTES", 1024)

    large_payload = b"A" * 2048
    files = {"image": ("large.jpg", large_payload, "image/jpeg")}
    response = client.post("/api/v1/recognize", files=files, data={"mode": "all"})
    assert response.status_code == 413
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "FILE_TOO_LARGE"


def test_custom_request_id_header(client):
    """Test custom X-Request-ID sent by client is propagated to response and logs."""
    custom_id = "req-client-custom-12345"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
