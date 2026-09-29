"""Convolutional Neural Network (CNN) architecture definition for CIFAR-10 classification."""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src.config import (
    INITIAL_LEARNING_RATE,
    INPUT_SHAPE,
    NUM_CLASSES,
)


def build_cifar10_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = INITIAL_LEARNING_RATE,
) -> tf.keras.Model:
    """Build, construct, and compile a modular CNN model for CIFAR-10 classification.

    CNN Architectural Foundations:
    -----------------------------
    1. Conv2D Layers: Apply learnable spatial filter kernels (3x3) across receptive fields
       to extract hierarchical visual features (edges -> textures -> parts -> objects).
    2. Batch Normalization: Re-centers and re-scales intermediate feature activations,
       drastically stabilizing internal covariate shifts and enabling higher learning rates.
    3. ReLU Activation: f(x) = max(0, x) provides non-linearity while overcoming vanishing
       gradients during backpropagation.
    4. MaxPooling2D: Downsamples spatial dimensions by 2x (32x32 -> 16x16 -> 8x8 -> 4x4),
       reducing parameter footprint and conferring spatial translation invariance.
    5. Dropout: Randomly zeros out a fraction of activations during training to prevent
       co-adaptation of feature detectors, providing strong regularization against overfitting.
    6. Dense + Softmax: Fully connected layers map high-level latent representations to
       normalized class probability distributions where sum(p_i) == 1.0.

    Args:
        input_shape: 3D tuple (height, width, channels), defaults to (32, 32, 3).
        num_classes: Number of output target classes, defaults to 10.
        learning_rate: Initial Adam optimizer learning rate.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    model = models.Sequential(name="cifar10_custom_cnn")

    # -------------------------------------------------------------------------
    # Input Layer
    # -------------------------------------------------------------------------
    model.add(layers.Input(shape=input_shape, name="input_image"))

    # -------------------------------------------------------------------------
    # Block 1: Low-level features (edges, color gradients, corners)
    # Output spatial resolution: 32x32 -> 16x16
    # -------------------------------------------------------------------------
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"))
    model.add(layers.BatchNormalization(name="bn1_1"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"))
    model.add(layers.BatchNormalization(name="bn1_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.2, name="dropout1"))

    # -------------------------------------------------------------------------
    # Block 2: Mid-level features (textures, motifs, geometric patterns)
    # Output spatial resolution: 16x16 -> 8x8
    # -------------------------------------------------------------------------
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"))
    model.add(layers.BatchNormalization(name="bn2_1"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"))
    model.add(layers.BatchNormalization(name="bn2_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.3, name="dropout2"))

    # -------------------------------------------------------------------------
    # Block 3: High-level semantic features (object parts, wheels, wings, fur)
    # Output spatial resolution: 8x8 -> 4x4
    # -------------------------------------------------------------------------
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"))
    model.add(layers.BatchNormalization(name="bn3_1"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"))
    model.add(layers.BatchNormalization(name="bn3_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.4, name="dropout3"))

    # -------------------------------------------------------------------------
    # Classification Head
    # -------------------------------------------------------------------------
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(128, activation="relu", name="dense_features"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.5, name="dropout_dense"))
    model.add(layers.Dense(num_classes, activation="softmax", name="classification_output"))

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def print_model_architecture(model: tf.keras.Model) -> None:
    """Print a clean visual summary of the model layer topology and parameters."""
    model.summary()
