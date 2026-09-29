"""Benchmarking and Verification Suite for Phase 6A Open-Vocabulary Engine."""

import json
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np

from src.phase6.config import (
    DEFAULT_PROMPT_TEMPLATE,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_NUM_THREADS,
    OPENCLIP_PRETRAINED,
)
from src.phase6.engine import OpenVocabEngine
from src.phase6.open_vocab_classifier import OpenVocabClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase6.benchmark")


def run_benchmark() -> dict:
    """Execute latency scaling, semantic separation, and negative control benchmarks."""
    logger.info("Initializing OpenVocabEngine for Phase 6A Benchmark...")
    engine = OpenVocabEngine(lazy_load=False)

    # 1. Prepare Test Images
    rose_path = Path("data/phase2/dataset_cache/flower_photos/flower_photos/roses/10090824183_d02c613f10_m.jpg")
    sunflower_path = Path("data/phase2/dataset_cache/flower_photos/flower_photos/sunflowers/1008566138_6927679c8a.jpg")
    daisy_path = Path("data/phase2/dataset_cache/flower_photos/flower_photos/daisy/100080576_f52e8ee070_m.jpg")

    images = {}
    if rose_path.exists():
        images["rose"] = Image.open(rose_path).convert("RGB")
    else:
        images["rose"] = Image.new("RGB", (224, 224), color=(220, 20, 30))

    if sunflower_path.exists():
        images["sunflower"] = Image.open(sunflower_path).convert("RGB")
    else:
        images["sunflower"] = Image.new("RGB", (224, 224), color=(240, 210, 20))

    if daisy_path.exists():
        images["daisy"] = Image.open(daisy_path).convert("RGB")
    else:
        images["daisy"] = Image.new("RGB", (224, 224), color=(250, 250, 250))

    # 2. Benchmark Query Count Latency Scaling (1, 5, 10, 20 queries)
    query_pools = {
        "1_query": ["rose"],
        "5_queries": ["rose", "sunflower", "tulip", "automobile", "laptop"],
        "10_queries": [
            "rose", "sunflower", "tulip", "daisy", "dandelion",
            "sports car", "laptop computer", "smartphone", "coffee mug", "airplane"
        ],
        "20_queries": [
            "rose", "sunflower", "tulip", "daisy", "dandelion",
            "sports car", "laptop computer", "smartphone", "coffee mug", "airplane",
            "golden retriever dog", "tabby cat", "horse", "elephant", "bicycle",
            "pizza", "guitar", "bookshelf", "waterfall", "swimming pool"
        ],
    }

    latency_results = {}
    test_img = images["rose"]

    for pool_name, queries in query_pools.items():
        # Warmup
        _ = engine.recognize(test_img, text_queries=queries, top_k=5)

        # 3 timed trials
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = engine.recognize(test_img, text_queries=queries, top_k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        avg_latency = float(np.mean(times))
        latency_results[pool_name] = {
            "query_count": len(queries),
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(float(np.min(times)), 2),
            "max_latency_ms": round(float(np.max(times)), 2),
            "top_match": res["open_vocabulary"]["top_match"]["query"],
            "top_score": res["open_vocabulary"]["top_match"]["similarity_score"],
        }
        logger.info(f"Query scaling [{pool_name} ({len(queries)} queries)]: {avg_latency:.1f}ms (Top match: {res['open_vocabulary']['top_match']['query']} @ {res['open_vocabulary']['top_match']['similarity_score']})")

    # 3. Semantic Discrimination Benchmark Across 3 Flower Classes
    discrimination_results = {}
    flower_candidates = ["rose", "sunflower", "daisy", "sports car", "laptop", "airplane"]

    for name, img in images.items():
        res = engine.recognize(img, text_queries=flower_candidates, top_k=6)
        top = res["open_vocabulary"]["top_match"]
        all_ranks = res["open_vocabulary"]["results"]
        discrimination_results[name] = {
            "expected": name,
            "top_match": top["query"],
            "top_score": top["similarity_score"],
            "is_correct": top["query"] == name,
            "all_ranked": all_ranks,
        }
        logger.info(f"Discrimination [{name}]: Top match '{top['query']}' (score: {top['similarity_score']:.4f}) - Correct: {top['query'] == name}")

    # 4. Negative Controls / Distractor Test (Image of Rose tested with only non-flower queries)
    distractor_queries = ["refrigerator", "submarine", "space shuttle", "microwave", "bulldozer"]
    distractor_res = engine.recognize(images["rose"], text_queries=distractor_queries, top_k=5)
    distractor_scores = [item["similarity_score"] for item in distractor_res["open_vocabulary"]["results"]]

    negative_control = {
        "distractor_queries": distractor_queries,
        "max_distractor_score": max(distractor_scores),
        "mean_distractor_score": round(float(np.mean(distractor_scores)), 4),
        "rose_true_score": discrimination_results["rose"]["top_score"],
        "semantic_margin": round(discrimination_results["rose"]["top_score"] - max(distractor_scores), 4),
    }
    logger.info(f"Negative Control: Rose true score ({negative_control['rose_true_score']:.4f}) vs Max distractor ({negative_control['max_distractor_score']:.4f}) -> Margin: +{negative_control['semantic_margin']:.4f}")

    benchmark_report = {
        "model": OPENCLIP_MODEL_NAME,
        "pretrained": OPENCLIP_PRETRAINED,
        "threads": OPENCLIP_NUM_THREADS,
        "latency_scaling": latency_results,
        "discrimination": discrimination_results,
        "negative_control": negative_control,
    }

    # Save benchmark report to models/phase6/benchmark_results.json
    out_path = Path("models/phase6/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info(f"Benchmark results saved to {out_path}")
    return benchmark_report


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report, indent=2))
