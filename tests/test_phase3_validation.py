"""Unit tests for Phase 3 image validation and normalization pipeline."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.phase3.image_validator import validate_and_load_image


def test_valid_jpg_validation():
    """Verify standard JPG image loading and metadata extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_sample.jpg"
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img.save(img_path, format="JPEG")

        res = validate_and_load_image(img_path)
        assert res.success is True
        assert res.image is not None
        assert res.metadata is not None
        assert res.metadata.width == 640
        assert res.metadata.height == 480
        assert res.metadata.channels == 3
        assert res.metadata.aspect_ratio == pytest.approx(640 / 480, 0.01)
        assert res.error is None


def test_valid_png_validation():
    """Verify PNG image loading and dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_sample.png"
        img = Image.new("RGB", (800, 600), color=(50, 200, 100))
        img.save(img_path, format="PNG")

        res = validate_and_load_image(img_path)
        assert res.success is True
        assert res.metadata.width == 800
        assert res.metadata.height == 600


def test_grayscale_to_rgb_conversion():
    """Verify grayscale single-channel images are safely converted to 3-channel RGB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "gray_sample.png"
        gray_arr = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        img = Image.fromarray(gray_arr, mode="L")
        img.save(img_path)

        res = validate_and_load_image(img_path)
        assert res.success is True
        assert res.image.mode == "RGB"
        assert res.np_array.shape == (100, 100, 3)


def test_rgba_to_rgb_compositing():
    """Verify RGBA transparent images are safely composited without black background artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "alpha_sample.png"
        img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        img.save(img_path)

        res = validate_and_load_image(img_path)
        assert res.success is True
        assert res.image.mode == "RGB"
        assert res.metadata.channels == 3


def test_missing_file_handling():
    """Verify non-existent paths return structured errors without crashing."""
    res = validate_and_load_image("non_existent_file_path_12345.jpg")
    assert res.success is False
    assert res.error_code == "FILE_NOT_FOUND"
    assert "does not exist" in res.error.lower()


def test_empty_zero_byte_file():
    """Verify 0-byte files return clear structured errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = Path(tmpdir) / "empty.jpg"
        empty_file.touch()

        res = validate_and_load_image(empty_file)
        assert res.success is False
        assert res.error_code == "EMPTY_FILE"


def test_unsupported_extension_handling():
    """Verify unsupported file extensions are rejected gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = Path(tmpdir) / "document.txt"
        txt_file.write_text("not an image")

        res = validate_and_load_image(txt_file)
        assert res.success is False
        assert res.error_code == "UNSUPPORTED_FORMAT"


def test_numpy_array_input_validation():
    """Verify in-memory NumPy array validation."""
    arr = np.random.randint(0, 256, size=(150, 150, 3), dtype=np.uint8)
    res = validate_and_load_image(arr)
    assert res.success is True
    assert res.metadata.width == 150
    assert res.metadata.height == 150
