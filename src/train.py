"""Training pipeline for CIFAR-10 image classifier."""

import argparse
import json
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INITIAL_LEARNING_RATE,
    INPUT_SHAPE,
    METADATA_SAVE_PATH,
    MIN_LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    PLOTS_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TRAINING_HISTORY_PLOT_PATH,
    TRAINING_METRICS_PATH,
    ensure_directories,
    set_reproducible_seed,
)
from src.data_loader import load_cifar10_dataset, save_dataset_samples
from src.model import build_cifar10_cnn
from src.preprocessing import create_tf_dataset, normalize_images


def plot_and_save_training_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Plot training and validation Loss & Accuracy curves and save to disk."""
    history_dict = history.history
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    ax1.plot(epochs_range, history_dict["accuracy"], "o-", label="Train Accuracy", color="#2563EB", linewidth=2)
    ax1.plot(epochs_range, history_dict["val_accuracy"], "s--", label="Val Accuracy", color="#10B981", linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss subplot
    ax1_loss = history_dict["loss"]
    ax1_val_loss = history_dict["val_loss"]
    ax2.plot(epochs_range, ax1_loss, "o-", label="Train Loss", color="#DC2626", linewidth=2)
    ax2.plot(epochs_range, ax1_val_loss, "s--", label="Val Loss", color="#F59E0B", linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = INITIAL_LEARNING_RATE,
    augment: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    """Execute the full training pipeline.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial Adam learning rate.
        augment: Whether to apply data augmentation during training.

    Returns:
        Dictionary with training summary metadata and history.
    """
    start_time = time.time()
    ensure_directories()
    set_reproducible_seed(RANDOM_SEED)

    print("=" * 70)
    print("           CIFAR-10 IMAGE RECOGNITION TRAINING PIPELINE           ")
    print("=" * 70)
    print(f"[CONFIG] Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print(f"[CONFIG] Augmentation: {augment} | Random Seed: {RANDOM_SEED}")

    # 1. Load Dataset Splits
    splits = load_cifar10_dataset()

    # Save visual sample grid
    save_dataset_samples(splits.x_train, splits.y_train, num_samples_per_class=2)

    # 2. Construct memory-efficient tf.data pipelines (with on-the-fly batch normalization)
    print("[INFO] Constructing streaming tf.data pipelines...")
    train_dataset = create_tf_dataset(
        splits.x_train,
        splits.y_train,
        batch_size=batch_size,
        is_training=True,
        augment=augment,
    )
    val_dataset = create_tf_dataset(
        splits.x_val,
        splits.y_val,
        batch_size=batch_size,
        is_training=False,
        augment=False,
    )

    # 4. Build or Resume Model
    if resume and MODEL_SAVE_PATH.exists():
        print(f"[INFO] Resuming training from existing checkpoint: {MODEL_SAVE_PATH}...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    else:
        print("[INFO] Building CNN Architecture from scratch...")
        model = build_cifar10_cnn(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
        )
    model.summary()

    # 5. Callbacks
    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    # 6. Train Model
    print("[INFO] Commencing model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callback_list,
        verbose=1,
    )

    training_duration = time.time() - start_time
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_val_loss = float(np.min(history.history["val_loss"]))
    final_train_acc = float(history.history["accuracy"][-1])
    final_train_loss = float(history.history["loss"][-1])

    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETED                    ")
    print("=" * 70)
    print(f"  * Training Duration:      {training_duration:.2f} seconds")
    print(f"  * Best Validation Acc:    {best_val_acc * 100:.2f}%")
    print(f"  * Best Validation Loss:   {best_val_loss:.4f}")
    print(f"  * Final Train Accuracy:   {final_train_acc * 100:.2f}%")
    print(f"  * Saved Best Model Path:  {MODEL_SAVE_PATH}")
    print("=" * 70)

    # 7. Save Plots & Metrics
    plot_and_save_training_history(history, str(TRAINING_HISTORY_PLOT_PATH))

    # Convert history numpy floats to python floats for JSON serialization
    serialized_history = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized_history, f, indent=2)
    print(f"[INFO] Training history logs saved to: {TRAINING_METRICS_PATH}")

    # 8. Save Metadata
    metadata = {
        "model_type": "CNN_CIFAR10_Custom",
        "dataset": "CIFAR-10",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_shape": list(INPUT_SHAPE),
        "hyperparameters": {
            "epochs_configured": epochs,
            "epochs_run": len(history.epoch),
            "batch_size": batch_size,
            "initial_learning_rate": learning_rate,
            "augmentation_enabled": augment,
            "random_seed": RANDOM_SEED,
        },
        "metrics": {
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "final_train_accuracy": final_train_acc,
            "final_train_loss": final_train_loss,
            "training_duration_seconds": round(training_duration, 2),
        },
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Model metadata saved to: {METADATA_SAVE_PATH}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN Image Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=INITIAL_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint in models/")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment=not args.no_augment,
        resume=args.resume,
    )
