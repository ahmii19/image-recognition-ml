"""Comprehensive evaluation, latency benchmarking, and out-of-domain analysis for Phase 2 models."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.phase2_config import (
    CLASS_NAMES,
    EFFICIENTNET_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    NUM_CLASSES,
    PHASE2_CONFUSION_MATRIX_PATH,
    PHASE2_METRICS_PATH,
    PHASE2_PLOTS_DIR,
    REAL_WORLD_PREDICTIONS_PATH,
    SELECTED_MODEL_PATH,
    ensure_phase2_directories,
)
from src.phase2_data import (
    create_phase2_splits,
    create_phase2_tf_dataset,
    download_and_prepare_dataset,
    get_image_file_paths_and_labels,
    preprocess_phase2_image,
)


def compute_top_k_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 5) -> float:
    """Compute Top-K categorical accuracy."""
    k = min(k, y_pred_probs.shape[1])
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = 0
    for true_label, top_preds in zip(y_true, top_k_preds):
        if true_label in top_preds:
            correct += 1
    return float(correct / len(y_true))


def measure_inference_latency(model: tf.keras.Model, num_iterations: int = 50) -> float:
    """Measure single-image inference latency in milliseconds on CPU/GPU."""
    dummy_input = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_input, verbose=0)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = model.predict(dummy_input, verbose=0)
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_iterations) * 1000.0
    return float(avg_latency_ms)


def plot_model_comparison_bar_chart(metrics_dict: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Plot multi-metric side-by-side comparison chart between Phase 1 and Phase 2 models."""
    models = list(metrics_dict.keys())
    top1_accs = [metrics_dict[m].get("top1_accuracy", 0.0) * 100 for m in models]
    f1_scores = [metrics_dict[m].get("macro_f1", 0.0) * 100 for m in models]
    latencies = [metrics_dict[m].get("latency_ms", 0.0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(models))
    width = 0.35

    # Subplot 1: Accuracy & F1
    ax1.bar(x - width/2, top1_accs, width, label="Top-1 Accuracy (%)", color="#2563EB")
    ax1.bar(x + width/2, f1_scores, width, label="Macro F1-Score (%)", color="#10B981")
    ax1.set_ylabel("Percentage (%)", fontsize=10)
    ax1.set_title("Model Accuracy & F1 Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Inference Latency
    ax2.bar(models, latencies, color="#F59E0B", width=0.4)
    ax2.set_ylabel("Latency per image (ms)", fontsize=10)
    ax2.set_title("Inference Speed Benchmark (Lower is Faster)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    for i, v in enumerate(latencies):
        ax2.text(i, v + 1, f"{v:.1f}ms", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparison chart saved to: {save_path}")


def evaluate_phase2_model(
    model: tf.keras.Model,
    test_paths: List[str],
    test_labels: List[int],
    model_family: str = "mobilenet_v2",
) -> Dict[str, Any]:
    """Evaluate a single Phase 2 model on the test dataset."""
    test_ds = create_phase2_tf_dataset(
        test_paths, test_labels, model_family=model_family, batch_size=32, is_training=False, augment=False
    )
    y_true = np.array(test_labels)

    # 1. Evaluate Loss & Loss
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # 2. Predict Probabilities
    y_probs = model.predict(test_ds, verbose=0)
    y_preds = np.argmax(y_probs, axis=1)

    # 3. Top-1 & Top-5 Accuracies
    top1_acc = float(test_acc)
    top5_acc = compute_top_k_accuracy(y_true, y_probs, k=5)

    # 4. Precision, Recall, F1
    report = classification_report(y_true, y_preds, target_names=CLASS_NAMES, output_dict=True, digits=4)

    # 5. Measure Latency
    latency_ms = measure_inference_latency(model)

    return {
        "test_loss": float(test_loss),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "latency_ms": latency_ms,
        "detailed_report": report,
        "y_probs": y_probs,
        "y_preds": y_preds,
    }


def run_phase2_evaluation() -> Dict[str, Any]:
    """Execute complete evaluation suite comparing MobileNetV2, EfficientNetB0, and baseline."""
    ensure_phase2_directories()

    print("=" * 75)
    print("             PHASE 2 MODEL EVALUATION & BENCHMARKING               ")
    print("=" * 75)

    dataset_dir = download_and_prepare_dataset()
    file_paths, labels, _ = get_image_file_paths_and_labels(dataset_dir)
    splits = create_phase2_splits(file_paths, labels)
    test_paths, test_labels = splits["test"]

    comparison_results: Dict[str, Any] = {}

    # 1. Evaluate MobileNetV2
    if MOBILENET_MODEL_PATH.exists():
        print("\n[INFO] Evaluating MobileNetV2 on 551 held-out test images...")
        mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
        mb_eval = evaluate_phase2_model(mobilenet_model, test_paths, test_labels, model_family="mobilenet_v2")
        comparison_results["MobileNetV2"] = {
            "top1_accuracy": mb_eval["top1_accuracy"],
            "top5_accuracy": mb_eval["top5_accuracy"],
            "test_loss": mb_eval["test_loss"],
            "macro_precision": mb_eval["macro_precision"],
            "macro_recall": mb_eval["macro_recall"],
            "macro_f1": mb_eval["macro_f1"],
            "latency_ms": mb_eval["latency_ms"],
            "total_params": int(mobilenet_model.count_params()),
            "file_size_mb": round(MOBILENET_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
        }

    # 2. Evaluate EfficientNetB0
    if EFFICIENTNET_MODEL_PATH.exists():
        print("\n[INFO] Evaluating EfficientNetB0 on 551 held-out test images...")
        eff_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
        eff_eval = evaluate_phase2_model(eff_model, test_paths, test_labels, model_family="efficientnet_b0")
        comparison_results["EfficientNetB0"] = {
            "top1_accuracy": eff_eval["top1_accuracy"],
            "top5_accuracy": eff_eval["top5_accuracy"],
            "test_loss": eff_eval["test_loss"],
            "macro_precision": eff_eval["macro_precision"],
            "macro_recall": eff_eval["macro_recall"],
            "macro_f1": eff_eval["macro_f1"],
            "latency_ms": eff_eval["latency_ms"],
            "total_params": int(eff_model.count_params()),
            "file_size_mb": round(EFFICIENTNET_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
        }

    # 3. Add Phase 1 Baseline for context
    comparison_results["Phase 1 CNN (Baseline)"] = {
        "top1_accuracy": 0.7935,
        "top5_accuracy": 0.9870,
        "test_loss": 0.6269,
        "macro_precision": 0.8053,
        "macro_recall": 0.7935,
        "macro_f1": 0.7907,
        "latency_ms": 12.4,
        "total_params": 552010,
        "file_size_mb": 6.74,
    }

    # 4. Generate comparison bar chart
    comparison_plot_path = PHASE2_PLOTS_DIR / "model_comparison.png"
    plot_model_comparison_bar_chart(comparison_results, comparison_plot_path)

    # 5. Save Confusion Matrix for Selected Model
    selected_model_path = SELECTED_MODEL_PATH if SELECTED_MODEL_PATH.exists() else MOBILENET_MODEL_PATH
    if selected_model_path.exists():
        sel_model = tf.keras.models.load_model(selected_model_path)
        sel_eval = evaluate_phase2_model(sel_model, test_paths, test_labels)
        cm = confusion_matrix(test_labels, sel_eval["y_preds"])

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(CLASS_NAMES)),
            yticks=np.arange(len(CLASS_NAMES)),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            title="Phase 2 Transfer Learning Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
                )
        plt.tight_layout()
        plt.savefig(str(PHASE2_CONFUSION_MATRIX_PATH), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Confusion Matrix saved to: {PHASE2_CONFUSION_MATRIX_PATH}")

    # 6. Save Metrics JSON
    with open(PHASE2_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"[INFO] Comparison metrics saved to: {PHASE2_METRICS_PATH}")

    # 7. Print Terminal Summary Table
    print("\n" + "=" * 80)
    print("                    FINAL COMPARATIVE BENCHMARK SUMMARY                    ")
    print("=" * 80)
    print(f"{'Model Name':<24} | {'Top-1 Acc':<9} | {'Top-5 Acc':<9} | {'F1-Score':<8} | {'Params':<9} | {'Latency':<8}")
    print("-" * 80)
    for model_name, m in comparison_results.items():
        print(
            f"{model_name:<24} | {m['top1_accuracy']*100:>8.2f}% | {m['top5_accuracy']*100:>8.2f}% | "
            f"{m['macro_f1']*100:>7.2f}% | {m['total_params']:>9,} | {m['latency_ms']:>6.1f}ms"
        )
    print("=" * 80)

    return comparison_results


if __name__ == "__main__":
    run_phase2_evaluation()
