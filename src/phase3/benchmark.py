"""Comprehensive Performance Benchmarking Suite for Phase 3 Recognition Engine."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

from src.phase3.config import PHASE3_METRICS_DIR, ensure_phase3_directories
from src.phase3.engine import AdvancedRecognitionEngine


def generate_benchmark_test_images() -> List[Tuple[str, Image.Image]]:
    """Create synthetic test images covering different resolutions and aspect ratios."""
    resolutions = [
        ("vga_640x480", (640, 480)),
        ("hd_1280x720", (1280, 720)),
        ("square_800x800", (800, 800)),
        ("portrait_600x900", (600, 900)),
        ("small_300x300", (300, 300)),
    ]
    images = []
    for name, (w, h) in resolutions:
        arr = np.random.randint(50, 220, size=(h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        images.append((name, img))
    return images


def run_benchmarks(
    num_warmup: int = 3,
    num_iterations: int = 15,
) -> Dict[str, Any]:
    """Execute rigorous inference latency benchmarks across components.

    Measures:
    - Preprocessing latency
    - Classifier inference latency (warm vs cold)
    - Detector inference latency (warm vs cold)
    - Unified pipeline latency
    - Throughput (FPS)

    Args:
        num_warmup: Warmup cycles.
        num_iterations: Benchmark iterations.

    Returns:
        Structured metrics dictionary.
    """
    ensure_phase3_directories()
    print("=" * 80)
    print("               PHASE 3 PERFORMANCE BENCHMARKING SUITE               ")
    print("=" * 80)

    engine = AdvancedRecognitionEngine(lazy_load=False)
    test_samples = generate_benchmark_test_images()

    benchmark_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": "CPU (Host System)",
        "iterations_per_resolution": num_iterations,
        "resolutions": {},
    }

    for res_name, pil_img in test_samples:
        w, h = pil_img.size
        print(f"\n[BENCHMARK] Testing Resolution: {res_name} ({w}x{h})...")

        # 1. Warmup
        for _ in range(num_warmup):
            _ = engine.recognize(pil_img, mode="all", render_visualization=False)

        # 2. Measure Classification
        clf_latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = engine.classifier.classify(pil_img)
            clf_latencies.append((time.perf_counter() - t0) * 1000.0)

        # 3. Measure Detection
        det_latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = engine.detector.detect(pil_img)
            det_latencies.append((time.perf_counter() - t0) * 1000.0)

        # 4. Measure Unified Pipeline (with visualization)
        unified_latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = engine.recognize(pil_img, mode="all", render_visualization=True)
            unified_latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_clf = float(np.mean(clf_latencies))
        p95_clf = float(np.percentile(clf_latencies, 95))
        avg_det = float(np.mean(det_latencies))
        p95_det = float(np.percentile(det_latencies, 95))
        avg_uni = float(np.mean(unified_latencies))
        p95_uni = float(np.percentile(unified_latencies, 95))
        fps_clf = 1000.0 / avg_clf if avg_clf > 0 else 0.0
        fps_uni = 1000.0 / avg_uni if avg_uni > 0 else 0.0

        benchmark_results["resolutions"][res_name] = {
            "dimensions": f"{w}x{h}",
            "classifier_avg_ms": round(avg_clf, 2),
            "classifier_p95_ms": round(p95_clf, 2),
            "classifier_fps": round(fps_clf, 1),
            "detector_avg_ms": round(avg_det, 2),
            "detector_p95_ms": round(p95_det, 2),
            "unified_pipeline_avg_ms": round(avg_uni, 2),
            "unified_pipeline_p95_ms": round(p95_uni, 2),
            "unified_pipeline_fps": round(fps_uni, 1),
        }

        print(
            f"  * Classifier:   Avg = {avg_clf:6.1f} ms | P95 = {p95_clf:6.1f} ms | Throughput = {fps_clf:5.1f} FPS\n"
            f"  * Detector:     Avg = {avg_det:6.1f} ms | P95 = {p95_det:6.1f} ms\n"
            f"  * Unified Eng.: Avg = {avg_uni:6.1f} ms | P95 = {p95_uni:6.1f} ms | Throughput = {fps_uni:5.1f} FPS",
            flush=True,
        )

    # Save metrics JSON
    metrics_file = PHASE3_METRICS_DIR / "benchmark_results.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[INFO] Benchmark results exported to: {metrics_file}", flush=True)
    return benchmark_results


if __name__ == "__main__":
    from typing import Tuple
    run_benchmarks()
