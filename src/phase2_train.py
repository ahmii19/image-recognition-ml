"""2-Stage Transfer Learning training and comparative evaluation pipeline for Phase 2."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.config import set_reproducible_seed
from src.phase2_config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_NAME,
    EFFICIENTNET_MODEL_PATH,
    FEATURE_EXTRACT_EPOCHS,
    FEATURE_EXTRACT_LR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MOBILENET_MODEL_PATH,
    NUM_CLASSES,
    PHASE2_DATA_DIR,
    PHASE2_METADATA_PATH,
    PHASE2_TRAINING_PLOT_PATH,
    RANDOM_SEED,
    SELECTED_MODEL_PATH,
    ensure_phase2_directories,
)
from src.phase2_data import (
    create_phase2_splits,
    create_phase2_tf_dataset,
    download_and_prepare_dataset,
    get_image_file_paths_and_labels,
)
from src.phase2_models import (
    build_efficientnetb0_model,
    build_mobilenetv2_model,
    get_model_parameter_counts,
    unfreeze_efficientnetb0_for_finetuning,
    unfreeze_mobilenetv2_for_finetuning,
)


def plot_combined_history(
    history_a: tf.keras.callbacks.History,
    history_b: tf.keras.callbacks.History,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot seamless Stage A (Feature Extraction) + Stage B (Fine-Tuning) curves."""
    acc = history_a.history["accuracy"] + history_b.history["accuracy"]
    val_acc = history_a.history["val_accuracy"] + history_b.history["val_accuracy"]
    loss = history_a.history["loss"] + history_b.history["loss"]
    val_loss = history_a.history["val_loss"] + history_b.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    split_epoch = len(history_a.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2563EB", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#10B981", linewidth=2)
    ax1.axvline(x=split_epoch, color="#EF4444", linestyle=":", label="Fine-Tuning Start")
    ax1.set_title(f"{model_name}: Accuracy (Stage A + Stage B)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#DC2626", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#F59E0B", linewidth=2)
    ax2.axvline(x=split_epoch, color="#EF4444", linestyle=":", label="Fine-Tuning Start")
    ax2.set_title(f"{model_name}: Loss (Stage A + Stage B)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training plot saved to: {save_path}")


def train_transfer_learning_model(
    model_type: str,
    splits: Dict[str, Tuple[list, list]],
    stage_a_epochs: int = FEATURE_EXTRACT_EPOCHS,
    stage_b_epochs: int = FINE_TUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Execute 2-Stage Transfer Learning on a specified backbone architecture."""
    start_time = time.time()
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]

    if model_type == "mobilenetv2":
        model_name = "MobileNetV2"
        checkpoint_path = MOBILENET_MODEL_PATH
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        family_name = "mobilenet_v2"
    elif model_type == "efficientnetb0":
        model_name = "EfficientNetB0"
        checkpoint_path = EFFICIENTNET_MODEL_PATH
        model = build_efficientnetb0_model(num_classes=NUM_CLASSES)
        family_name = "efficientnet_b0"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print("\n" + "=" * 75)
    print(f"       TRAINING BACKBONE: {model_name} (2-Stage Transfer Learning)       ")
    print("=" * 75)

    # 1. Create tf.data pipelines with backbone-specific preprocessing
    train_ds = create_phase2_tf_dataset(
        train_paths, train_labels, model_family=family_name, batch_size=batch_size, is_training=True, augment=True
    )
    val_ds = create_phase2_tf_dataset(
        val_paths, val_labels, model_family=family_name, batch_size=batch_size, is_training=False, augment=False
    )

    # =========================================================================
    # STAGE A: Feature Extraction (Backbone Frozen)
    # =========================================================================
    print(f"\n[STAGE A] Feature Extraction ({stage_a_epochs} Epochs, Backbone Frozen, LR={FEATURE_EXTRACT_LR})...")
    cb_stage_a = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ]

    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage_a_epochs,
        callbacks=cb_stage_a,
        verbose=1,
    )

    stage_a_val_acc = float(np.max(history_a.history["val_accuracy"]))
    print(f"[STAGE A RESULT] Best Validation Accuracy: {stage_a_val_acc * 100:.2f}%")

    # =========================================================================
    # STAGE B: Fine-Tuning (Top Backbone Blocks Unfrozen)
    # =========================================================================
    print(f"\n[STAGE B] Fine-Tuning ({stage_b_epochs} Epochs, Top Blocks Unfrozen, LR={FINE_TUNE_LR})...")
    if model_type == "mobilenetv2":
        model = unfreeze_mobilenetv2_for_finetuning(model, learning_rate=FINE_TUNE_LR)
    else:
        model = unfreeze_efficientnetb0_for_finetuning(model, learning_rate=FINE_TUNE_LR)

    cb_stage_b = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage_b_epochs,
        callbacks=cb_stage_b,
        verbose=1,
    )

    total_duration = time.time() - start_time
    best_val_acc = float(max(np.max(history_a.history["val_accuracy"]), np.max(history_b.history["val_accuracy"])))
    best_val_loss = float(min(np.min(history_a.history["val_loss"]), np.min(history_b.history["val_loss"])))

    # Save training curves
    plot_path = PHASE2_TRAINING_PLOT_PATH.parent / f"{model_type}_training_history.png"
    plot_combined_history(history_a, history_b, model_name, plot_path)

    param_info = get_model_parameter_counts(model)
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0.0

    summary = {
        "model_type": model_type,
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "stage_a_epochs": len(history_a.epoch),
        "stage_b_epochs": len(history_b.epoch),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "duration_seconds": round(total_duration, 2),
        "params": param_info,
        "file_size_mb": round(file_size_mb, 2),
    }

    return model, summary


def run_phase2_training_pipeline(
    train_efficientnet: bool = True,
    stage_a_epochs: int = FEATURE_EXTRACT_EPOCHS,
    stage_b_epochs: int = FINE_TUNE_EPOCHS,
) -> Dict[str, Any]:
    """Train MobileNetV2 and EfficientNetB0, compare validation performance, and save metadata."""
    ensure_phase2_directories()
    set_reproducible_seed(RANDOM_SEED)

    print("=" * 75)
    print("      PHASE 2: REAL-WORLD TRANSFER LEARNING TRAINING PIPELINE        ")
    print("=" * 75)

    # 1. Download and Partition Real-World Dataset
    dataset_dir = download_and_prepare_dataset()
    file_paths, labels, discovered_classes = get_image_file_paths_and_labels(dataset_dir)
    splits = create_phase2_splits(file_paths, labels)

    # 2. Train MobileNetV2
    _, mobilenet_summary = train_transfer_learning_model(
        model_type="mobilenetv2",
        splits=splits,
        stage_a_epochs=stage_a_epochs,
        stage_b_epochs=stage_b_epochs,
    )

    models_comparison = {"mobilenetv2": mobilenet_summary}

    # 3. Train EfficientNetB0 (if enabled)
    if train_efficientnet:
        _, efficientnet_summary = train_transfer_learning_model(
            model_type="efficientnetb0",
            splits=splits,
            stage_a_epochs=stage_a_epochs,
            stage_b_epochs=stage_b_epochs,
        )
        models_comparison["efficientnetb0"] = efficientnet_summary

    # 4. Model Selection based on validation accuracy & efficiency
    best_candidate_key = max(models_comparison.keys(), key=lambda k: models_comparison[k]["best_val_accuracy"])
    selected_summary = models_comparison[best_candidate_key]
    selected_src_path = Path(selected_summary["checkpoint_path"])

    shutil.copyfile(selected_src_path, SELECTED_MODEL_PATH)
    print("\n" + "=" * 75)
    print("                    MODEL SELECTION SUMMARY                       ")
    print("=" * 75)
    for k, v in models_comparison.items():
        print(
            f"  * {v['model_name']:<15} -> Val Acc: {v['best_val_accuracy']*100:.2f}% | "
            f"Total Params: {v['params']['total_params']:,} | Size: {v['file_size_mb']:.2f} MB | Time: {v['duration_seconds']:.1f}s"
        )
    print(f"\n  ==> Selected Best Model: {selected_summary['model_name']} (Copied to {SELECTED_MODEL_PATH})")
    print("=" * 75)

    # 5. Export Phase 2 Metadata
    metadata = {
        "phase": 2,
        "dataset_name": DATASET_NAME,
        "dataset_classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "split_counts": {
            "train": len(splits["train"][0]),
            "val": len(splits["val"][0]),
            "test": len(splits["test"][0]),
        },
        "selected_model": best_candidate_key,
        "selected_model_path": str(SELECTED_MODEL_PATH),
        "models_evaluated": models_comparison,
    }

    with open(PHASE2_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Phase 2 Metadata exported to: {PHASE2_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Transfer Learning Training")
    parser.add_argument("--stage_a_epochs", type=int, default=FEATURE_EXTRACT_EPOCHS, help="Stage A epochs")
    parser.add_argument("--stage_b_epochs", type=int, default=FINE_TUNE_EPOCHS, help="Stage B epochs")
    parser.add_argument("--skip_efficientnet", action="store_true", help="Skip EfficientNet training")

    args = parser.parse_args()
    run_phase2_training_pipeline(
        train_efficientnet=not args.skip_efficientnet,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
    )
