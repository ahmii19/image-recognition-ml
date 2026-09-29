"""Unit tests for Phase 2 data preprocessing and partitioning."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.phase2_config import CLASS_NAMES, NUM_CLASSES
from src.phase2_data import create_phase2_splits, preprocess_phase2_image


def test_phase2_class_names_and_count():
    """Verify Phase 2 class definitions."""
    assert len(CLASS_NAMES) == NUM_CLASSES == 5
    assert "roses" in CLASS_NAMES
    assert "daisy" in CLASS_NAMES
    assert "sunflowers" in CLASS_NAMES


def test_create_phase2_splits_ratios():
    """Verify stratified partitioning produces exact 70/15/15 train/val/test splits."""
    # Synthetic dataset of 100 samples across 5 classes
    mock_paths = [f"img_{i}.jpg" for i in range(100)]
    mock_labels = [i % 5 for i in range(100)]

    splits = create_phase2_splits(
        mock_paths,
        mock_labels,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # Total count matches original dataset
    assert len(train_paths) + len(val_paths) + len(test_paths) == 100

    # Ratios approximately 70/15/15 within rounding tolerances of stratified partitioning
    assert abs(len(test_paths) - 15) <= 1
    assert abs(len(val_paths) - 15) <= 1
    assert abs(len(train_paths) - 70) <= 1

    # Verify no overlap between partitions
    assert len(set(train_paths).intersection(set(val_paths))) == 0
    assert len(set(train_paths).intersection(set(test_paths))) == 0
    assert len(set(val_paths).intersection(set(test_paths))) == 0


def test_preprocess_phase2_image_shape_and_aspect_ratio():
    """Verify single-image preprocessing produces (1, 224, 224, 3) tensor regardless of original dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create non-square high-res image (800x400)
        img_path = Path(tmpdir) / "high_res_sample.jpg"
        img = Image.new("RGB", (800, 400), color=(150, 75, 200))
        img.save(img_path)

        processed = preprocess_phase2_image(img_path, model_family="mobilenet_v2")

        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        # MobileNet normalization bounds are [-1.0, 1.0]
        assert np.all(processed >= -1.0)
        assert np.all(processed <= 1.0)
