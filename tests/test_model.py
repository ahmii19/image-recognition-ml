"""Unit tests for CNN architecture and forward pass output shapes."""

import numpy as np
import pytest
import tensorflow as tf

from src.config import INPUT_SHAPE, NUM_CLASSES
from src.model import build_cifar10_cnn


def test_model_build_and_shapes():
    """Verify CNN model compiles with expected input shape and 10-class softmax output."""
    model = build_cifar10_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 10)
    assert model.loss == "sparse_categorical_crossentropy"


def test_model_forward_pass():
    """Verify that a dummy batch propagates through the model producing valid probability distributions."""
    model = build_cifar10_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    # Create synthetic batch of 4 images
    batch_size = 4
    dummy_input = np.random.uniform(0.0, 1.0, size=(batch_size, 32, 32, 3)).astype(np.float32)

    predictions = model.predict(dummy_input, verbose=0)

    assert predictions.shape == (batch_size, 10)
    # Check that probabilities sum to 1.0 for each sample in batch
    sums = np.sum(predictions, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    # Check all probabilities are within [0.0, 1.0]
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
