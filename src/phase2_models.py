"""Transfer learning model definitions and unfreezing mechanisms for MobileNetV2 and EfficientNetB0."""

from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

from src.phase2_config import (
    EFFICIENTNET_FINE_TUNE_AT,
    FEATURE_EXTRACT_LR,
    FINE_TUNE_LR,
    INPUT_SHAPE,
    MOBILENET_FINE_TUNE_AT,
    NUM_CLASSES,
)


def build_mobilenetv2_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = FEATURE_EXTRACT_LR,
) -> tf.keras.Model:
    """Build a MobileNetV2 Transfer Learning Model (Stage A: Feature Extraction).

    Architecture:
    -------------
    1. Pretrained ImageNet MobileNetV2 base (weights frozen).
    2. GlobalAveragePooling2D reduces spatial tensor (7x7x1280) to feature vector (1280).
    3. Dense projection (256 units) + BatchNormalization + Dropout(0.4).
    4. Softmax output layer (num_classes).

    Args:
        num_classes: Number of target categories.
        input_shape: (224, 224, 3) input shape.
        learning_rate: Initial learning rate for feature extraction.

    Returns:
        Compiled tf.keras.Model ready for Stage A feature extraction.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False  # Freeze all base model layers

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_features")(x)
    x = layers.Dropout(0.4, name="dropout_features")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer_model")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_mobilenetv2_for_finetuning(
    model: tf.keras.Model,
    fine_tune_at: int = MOBILENET_FINE_TUNE_AT,
    learning_rate: float = FINE_TUNE_LR,
) -> tf.keras.Model:
    """Unfreeze top layers of MobileNetV2 backbone for Stage B fine-tuning.

    Args:
        model: Compiled transfer learning model.
        fine_tune_at: Layer index from which to begin unfreezing.
        learning_rate: Low learning rate for fine-tuning.

    Returns:
        Re-compiled model ready for fine-tuning.
    """
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 base layer in model.")

    base_model.trainable = True

    # Freeze all lower layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    print(
        f"[INFO] MobileNetV2 Fine-Tuning Configuration:\n"
        f"       - Total Base Layers:    {len(base_model.layers)}\n"
        f"       - Frozen Lower Layers:  {fine_tune_at}\n"
        f"       - Unfrozen Top Layers:  {len(base_model.layers) - fine_tune_at}\n"
        f"       - Fine-Tuning LR:       {learning_rate}"
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb0_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = FEATURE_EXTRACT_LR,
) -> tf.keras.Model:
    """Build an EfficientNetB0 Transfer Learning Model (Stage A: Feature Extraction).

    Architecture:
    -------------
    1. Pretrained ImageNet EfficientNetB0 base (weights frozen).
    2. GlobalAveragePooling2D reduces spatial tensor (7x7x1280) to feature vector (1280).
    3. Dense projection (256 units) + BatchNormalization + Dropout(0.4).
    4. Softmax output layer (num_classes).

    Args:
        num_classes: Number of target categories.
        input_shape: (224, 224, 3) input shape.
        learning_rate: Initial learning rate for feature extraction.

    Returns:
        Compiled tf.keras.Model ready for Stage A feature extraction.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False  # Freeze all base model layers

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_features")(x)
    x = layers.Dropout(0.4, name="dropout_features")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_transfer_model")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_efficientnetb0_for_finetuning(
    model: tf.keras.Model,
    fine_tune_at: int = EFFICIENTNET_FINE_TUNE_AT,
    learning_rate: float = FINE_TUNE_LR,
) -> tf.keras.Model:
    """Unfreeze top layers of EfficientNetB0 backbone for Stage B fine-tuning.

    Args:
        model: Compiled transfer learning model.
        fine_tune_at: Layer index from which to begin unfreezing.
        learning_rate: Low learning rate for fine-tuning.

    Returns:
        Re-compiled model ready for fine-tuning.
    """
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate EfficientNetB0 base layer in model.")

    base_model.trainable = True

    # Freeze all lower layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    print(
        f"[INFO] EfficientNetB0 Fine-Tuning Configuration:\n"
        f"       - Total Base Layers:    {len(base_model.layers)}\n"
        f"       - Frozen Lower Layers:  {fine_tune_at}\n"
        f"       - Unfrozen Top Layers:  {len(base_model.layers) - fine_tune_at}\n"
        f"       - Fine-Tuning LR:       {learning_rate}"
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model_parameter_counts(model: tf.keras.Model) -> Dict[str, int]:
    """Calculate trainable, non-trainable, and total parameter counts for a model."""
    trainable_count = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_count = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    total_count = trainable_count + non_trainable_count
    return {
        "trainable_params": trainable_count,
        "non_trainable_params": non_trainable_count,
        "total_params": total_count,
    }
