"""Evaluation pipeline for CIFAR-10 image classifier on the test dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    ensure_directories,
)
from src.data_loader import load_cifar10_dataset
from src.preprocessing import create_tf_dataset, normalize_images


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    title: str = "CIFAR-10 Test Confusion Matrix",
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:,}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to: {save_path}")


def evaluate_model(model_path: Path = MODEL_SAVE_PATH) -> Dict[str, Any]:
    """Evaluate the trained CNN model on the untouched CIFAR-10 test set.

    Args:
        model_path: Path to the saved Keras model file.

    Returns:
        Dictionary containing evaluation metrics.
    """
    ensure_directories()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please run 'python -m src.train' first."
        )

    print("=" * 70)
    print("           CIFAR-10 IMAGE RECOGNITION EVALUATION PIPELINE         ")
    print("=" * 70)
    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 1. Load untouched test set
    splits = load_cifar10_dataset()
    test_dataset = create_tf_dataset(
        splits.x_test,
        splits.y_test,
        batch_size=128,
        is_training=False,
        augment=False,
    )
    y_test = splits.y_test

    # 2. Evaluate Loss and Accuracy
    print("[INFO] Computing loss and accuracy on 10,000 test images...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    # 3. Predict class probabilities
    print("[INFO] Generating predictions for full test split...")
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 4. Compute Sklearn Classification Metrics
    report_dict = classification_report(
        y_test,
        y_pred_classes,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
    )
    report_text = classification_report(
        y_test,
        y_pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
    )

    # 5. Compute and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, CLASS_NAMES, str(CONFUSION_MATRIX_PLOT_PATH))

    # 6. Save Metrics Report to JSON
    summary_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_avg_precision": float(report_dict["macro avg"]["precision"]),
        "macro_avg_recall": float(report_dict["macro avg"]["recall"]),
        "macro_avg_f1_score": float(report_dict["macro avg"]["f1-score"]),
        "weighted_avg_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_avg_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_avg_f1_score": float(report_dict["weighted avg"]["f1-score"]),
        "detailed_classification_report": report_dict,
    }

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"[INFO] Classification report saved to: {CLASSIFICATION_REPORT_PATH}")

    # 7. Print Terminal Summary
    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS                    ")
    print("=" * 70)
    print(f"  * Test Loss:           {test_loss:.4f}")
    print(f"  * Test Accuracy:       {test_accuracy * 100:.2f}%")
    print(f"  * Macro Precision:     {summary_metrics['macro_avg_precision'] * 100:.2f}%")
    print(f"  * Macro Recall:        {summary_metrics['macro_avg_recall'] * 100:.2f}%")
    print(f"  * Macro F1-Score:      {summary_metrics['macro_avg_f1_score'] * 100:.2f}%")
    print("-" * 70)
    print("Detailed Classification Report:\n")
    print(report_text)
    print("=" * 70)

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 Image Classifier")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_SAVE_PATH),
        help="Path to trained .keras model",
    )
    args = parser.parse_args()
    evaluate_model(model_path=Path(args.model_path))
