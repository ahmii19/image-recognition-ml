"""Unit tests for preprocessing and inference pipeline."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.config import CLASS_NAMES, INPUT_SHAPE
from src.preprocessing import preprocess_single_image


def test_preprocess_single_image():
    """Verify that preprocess_single_image loads, resizes, and normalizes an image file correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an arbitrary 100x150 RGB image
        img_path = Path(tmpdir) / "test_img.jpg"
        img = Image.new("RGB", (100, 150), color=(200, 100, 50))
        img.save(img_path)

        processed = preprocess_single_image(str(img_path))

        assert processed.shape == (1, 32, 32, 3)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)


def test_preprocess_grayscale_conversion():
    """Verify that grayscale image input is automatically converted to 3-channel RGB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "gray_img.png"
        img = Image.new("L", (64, 64), color=128)
        img.save(img_path)

        processed = preprocess_single_image(str(img_path))

        assert processed.shape == (1, 32, 32, 3)
        assert processed.dtype == np.float32


def test_preprocess_nonexistent_file():
    """Verify that attempting to preprocess a missing file raises FileNotFoundError or ValueError."""
    with pytest.raises((FileNotFoundError, ValueError)):
        preprocess_single_image("non_existent_file_path_12345.jpg")
