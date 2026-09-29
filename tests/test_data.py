"""Unit tests for dataset loading, normalization, and class configuration."""

import numpy as np
import pytest

from src.config import CLASS_NAMES, NUM_CLASSES
from src.preprocessing import normalize_images


def test_class_names_configuration():
    """Verify that CIFAR-10 class definitions have exactly 10 distinct classes."""
    assert len(CLASS_NAMES) == NUM_CLASSES == 10
    assert len(set(CLASS_NAMES)) == 10
    assert "dog" in CLASS_NAMES
    assert "cat" in CLASS_NAMES
    assert "airplane" in CLASS_NAMES


def test_normalize_images_range_and_dtype():
    """Verify normalization scales uint8 [0..255] images to float32 [0.0..1.0]."""
    # Create synthetic test batch
    raw_images = np.array([
        [[[0, 128, 255]]],
        [[[255, 64, 0]]],
    ], dtype=np.uint8)

    normalized = normalize_images(raw_images)

    assert normalized.dtype == np.float32
    assert normalized.shape == raw_images.shape
    assert np.isclose(normalized[0, 0, 0, 0], 0.0)
    assert np.isclose(normalized[0, 0, 0, 1], 128.0 / 255.0)
    assert np.isclose(normalized[0, 0, 0, 2], 1.0)
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


def test_normalize_idempotence():
    """Verify that normalizing an already normalized float32 array is safe."""
    float_images = np.array([[[[0.5, 0.2, 0.9]]]], dtype=np.float32)
    normalized = normalize_images(float_images)
    assert np.allclose(float_images, normalized)
