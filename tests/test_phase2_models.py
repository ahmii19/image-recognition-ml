"""Unit tests for Phase 2 Transfer Learning architectures."""

import numpy as np
import pytest
import tensorflow as tf

from src.phase2_config import NUM_CLASSES
from src.phase2_models import (
    build_efficientnetb0_model,
    build_mobilenetv2_model,
    get_model_parameter_counts,
    unfreeze_mobilenetv2_for_finetuning,
)


def test_mobilenetv2_model_build_and_forward_pass():
    """Verify MobileNetV2 builds with frozen backbone and produces valid 5-class output."""
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES)

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, NUM_CLASSES)

    params_stage_a = get_model_parameter_counts(model)
    # In Stage A, base model weights are non-trainable
    assert params_stage_a["non_trainable_params"] > 2_000_000
    assert params_stage_a["trainable_params"] < 500_000

    # Test forward pass with dummy tensor
    dummy_input = np.random.uniform(-1.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape == (2, NUM_CLASSES)
    assert np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-4)


def test_mobilenetv2_unfreezing():
    """Verify unfreezing increases trainable parameters for Stage B fine-tuning."""
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
    params_a = get_model_parameter_counts(model)

    model = unfreeze_mobilenetv2_for_finetuning(model, fine_tune_at=100)
    params_b = get_model_parameter_counts(model)

    # Trainable parameters must increase significantly after unfreezing top layers
    assert params_b["trainable_params"] > params_a["trainable_params"]
