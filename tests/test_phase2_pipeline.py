"""Unit tests for Phase 2 inference formatting, Top-K accuracy, and uncertainty evaluation."""

import numpy as np
import pytest

from src.phase2_evaluate import compute_top_k_accuracy
from src.predict_phase2 import format_phase2_output


def test_compute_top_k_accuracy():
    """Verify Top-K accuracy computation on known ground truth and probability matrices."""
    y_true = np.array([0, 1, 2, 3])
    # Class 0: highest is 0 (correct)
    # Class 1: highest is 2, second is 1 (top-1 incorrect, top-2 correct)
    # Class 2: highest is 2 (correct)
    # Class 3: highest is 0, 1, 2, 4 (incorrect)
    y_probs = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.4, 0.5, 0.0],
        [0.1, 0.1, 0.7, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ])

    top1 = compute_top_k_accuracy(y_true, y_probs, k=1)
    top2 = compute_top_k_accuracy(y_true, y_probs, k=2)

    assert top1 == 0.50  # Samples 0 and 2 are top-1 correct (2/4 = 50%)
    assert top2 == 0.75  # Samples 0, 1, 2 are top-2 correct (3/4 = 75%)


def test_format_phase2_output_uncertainty():
    """Verify format_phase2_output includes warning when prediction is uncertain."""
    mock_res_confident = {
        "image_name": "flower.jpg",
        "mode": "transfer_learned",
        "prediction": "roses",
        "confidence_percent": 88.5,
        "is_uncertain": False,
        "uncertainty_threshold": 40.0,
        "top_k": [{"class": "roses", "confidence_percent": 88.5}],
    }
    output_conf = format_phase2_output(mock_res_confident, top_k=1)
    assert "UNCERTAINTY WARNING" not in output_conf

    mock_res_uncertain = {
        "image_name": "unknown_object.jpg",
        "mode": "transfer_learned",
        "prediction": "daisy",
        "confidence_percent": 28.2,
        "is_uncertain": True,
        "uncertainty_threshold": 40.0,
        "top_k": [{"class": "daisy", "confidence_percent": 28.2}],
    }
    output_unc = format_phase2_output(mock_res_uncertain, top_k=1)
    assert "UNCERTAINTY WARNING" in output_unc
