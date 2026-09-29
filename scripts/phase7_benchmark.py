"""Phase 7B — Isolated VLM Benchmark Script.

Benchmarks the Qwen2.5-VL-3B-Instruct General Image Understanding engine
across all supported PromptModes, with configurable number of warmup and
timed runs per image.

Usage (local CPU — slow but functional):
    python scripts/phase7_benchmark.py --image path/to/image.jpg

Usage (Google Colab T4 — recommended):
    !python scripts/phase7_benchmark.py --image /content/test_image.jpg --device cuda

Usage (all prompt modes, 3 runs each):
    python scripts/phase7_benchmark.py --image img.jpg --runs 3 --all-modes

IMPORTANT:
    - This script downloads Qwen2.5-VL-3B-Instruct on first run (~6.5 GB).
    - On CPU it is EXTREMELY slow (5–30+ minutes per image). Use Colab GPU.
    - Results are saved to outputs/phase7_benchmark_results.json.
    - This script does NOT modify any Phase 1–6 source files.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.device import get_device_info, resolve_device
from src.phase7.lifecycle import VLMLifecycleManager
from src.phase7.prompts import PromptMode, list_prompt_modes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7.benchmark")


# ── Benchmark Modes ───────────────────────────────────────────────────────────

ALL_BENCHMARK_MODES = [
    PromptMode.BRIEF,
    PromptMode.GENERAL,
    PromptMode.DETAILED,
    PromptMode.DOCUMENT,
    PromptMode.DIAGRAM,
    PromptMode.CHART,
    PromptMode.MAP,
    PromptMode.MEDICAL,
]


def _run_single_benchmark(
    image_path: str,
    mode: PromptMode,
    runs: int,
    warmup_runs: int,
) -> Dict[str, Any]:
    """Run a single prompt mode benchmark with warmup and timed iterations.

    Args:
        image_path: Path to the image file to benchmark.
        mode: The PromptMode to benchmark.
        runs: Number of timed inference runs.
        warmup_runs: Number of warmup runs before timing begins.

    Returns:
        Benchmark result dict for this mode.
    """
    logger.info("Benchmarking mode: %s (warmup=%d, runs=%d)", mode.value, warmup_runs, runs)

    engine = VLMLifecycleManager.get_engine()

    # Warmup
    for w in range(warmup_runs):
        logger.info("  Warmup %d/%d...", w + 1, warmup_runs)
        engine.understand(image_path, mode=mode)

    # Timed runs
    timings_ms: List[float] = []
    last_result: Optional[Dict[str, Any]] = None

    for r in range(runs):
        logger.info("  Timed run %d/%d...", r + 1, runs)
        t0 = time.perf_counter()
        result = engine.understand(image_path, mode=mode)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(elapsed_ms)
        last_result = result
        logger.info(
            "  Run %d: %.0f ms | success=%s | recovered=%s",
            r + 1, elapsed_ms,
            result.get("success"),
            result.get("parse_meta", {}).get("recovered"),
        )

    avg_ms = sum(timings_ms) / len(timings_ms) if timings_ms else 0.0
    min_ms = min(timings_ms) if timings_ms else 0.0
    max_ms = max(timings_ms) if timings_ms else 0.0

    return {
        "mode": mode.value,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "timings_ms": [round(t, 2) for t in timings_ms],
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "last_result_success": last_result.get("success") if last_result else False,
        "last_understanding": last_result.get("understanding", {}) if last_result else {},
        "last_parse_meta": last_result.get("parse_meta", {}) if last_result else {},
        "last_timing": last_result.get("timing", {}) if last_result else {},
    }


def run_benchmark(
    image_path: str,
    modes: List[PromptMode],
    runs: int,
    warmup_runs: int,
    output_path: Optional[str],
) -> Dict[str, Any]:
    """Execute the full benchmark suite.

    Args:
        image_path: Path to the image to analyse.
        modes: List of PromptModes to benchmark.
        runs: Timed iterations per mode.
        warmup_runs: Warmup iterations per mode (not timed).
        output_path: Optional JSON output path; if None, saves to outputs/ dir.

    Returns:
        Full benchmark report dict.
    """
    logger.info("=" * 60)
    logger.info("PHASE 7B — VLM BENCHMARK")
    logger.info("=" * 60)

    device_info = get_device_info()
    logger.info("Device: %s", device_info)

    engine_info = {}
    try:
        engine = VLMLifecycleManager.get_engine()
        engine_info = engine.get_info()
        logger.info("VLMEngine loaded: %s params on %s", engine_info.get("param_count"), engine_info.get("device"))
    except Exception as exc:
        logger.error("Failed to initialize VLMEngine: %s", exc)
        raise

    benchmark_start = time.perf_counter()
    mode_results: List[Dict[str, Any]] = []

    for mode in modes:
        try:
            mode_result = _run_single_benchmark(image_path, mode, runs, warmup_runs)
            mode_results.append(mode_result)
        except Exception as exc:
            logger.error("Benchmark failed for mode %s: %s", mode.value, exc)
            mode_results.append({
                "mode": mode.value,
                "error": str(exc),
                "runs": runs,
            })

    total_benchmark_ms = (time.perf_counter() - benchmark_start) * 1000.0

    # Compute summary stats across modes
    successful_modes = [m for m in mode_results if "avg_ms" in m]
    overall_avg_ms = (
        sum(m["avg_ms"] for m in successful_modes) / len(successful_modes)
        if successful_modes
        else 0.0
    )
    fastest_mode = min(successful_modes, key=lambda m: m["avg_ms"]) if successful_modes else {}
    slowest_mode = max(successful_modes, key=lambda m: m["avg_ms"]) if successful_modes else {}

    report = {
        "benchmark_id": f"phase7b_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        "runs_per_mode": runs,
        "warmup_runs": warmup_runs,
        "modes_tested": [m.value for m in modes],
        "device_info": device_info,
        "engine_info": engine_info,
        "total_benchmark_ms": round(total_benchmark_ms, 2),
        "summary": {
            "modes_tested": len(modes),
            "modes_succeeded": len(successful_modes),
            "overall_avg_ms": round(overall_avg_ms, 2),
            "fastest_mode": fastest_mode.get("mode", "N/A"),
            "fastest_avg_ms": fastest_mode.get("avg_ms", 0),
            "slowest_mode": slowest_mode.get("mode", "N/A"),
            "slowest_avg_ms": slowest_mode.get("avg_ms", 0),
        },
        "mode_results": mode_results,
    }

    # Save report
    if output_path is None:
        outputs_dir = _PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        output_path = str(outputs_dir / f"phase7_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    Path(output_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Benchmark report saved to: %s", output_path)

    # Print summary to console
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info("Modes tested:    %d", report["summary"]["modes_tested"])
    logger.info("Modes succeeded: %d", report["summary"]["modes_succeeded"])
    logger.info("Overall avg:     %.0f ms", report["summary"]["overall_avg_ms"])
    logger.info("Fastest mode:    %s (%.0f ms)", report["summary"]["fastest_mode"], report["summary"]["fastest_avg_ms"])
    logger.info("Slowest mode:    %s (%.0f ms)", report["summary"]["slowest_mode"], report["summary"]["slowest_avg_ms"])
    logger.info("Total time:      %.0f ms (%.1f min)", total_benchmark_ms, total_benchmark_ms / 60000)
    logger.info("Report saved to: %s", output_path)

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7B VLM General Image Understanding Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device: auto, cpu, cuda. Overrides ML_DEVICE env var.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of timed inference runs per mode.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per mode (not timed).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="general",
        choices=[m.value for m in PromptMode if m != PromptMode.CUSTOM],
        help="Single prompt mode to benchmark.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Benchmark all supported prompt modes (overrides --mode).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. Defaults to outputs/phase7_benchmark_<timestamp>.json.",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Print all available prompt modes and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_modes:
        print("\nAvailable Phase 7B Prompt Modes:\n")
        for entry in list_prompt_modes():
            print(f"  {entry['mode']:12s} — {entry['description']}")
        print()
        sys.exit(0)

    # Set device override via environment
    if args.device:
        import os
        os.environ["VLM_DEVICE"] = args.device
        os.environ["ML_DEVICE"] = args.device
        logger.info("Device overridden to: %s", args.device)

    # Validate image path
    image_path = args.image
    if not Path(image_path).exists():
        logger.error("Image file not found: %s", image_path)
        sys.exit(1)

    # Select modes
    if args.all_modes:
        modes = ALL_BENCHMARK_MODES
        logger.info("Benchmarking ALL %d prompt modes.", len(modes))
    else:
        modes = [PromptMode(args.mode)]
        logger.info("Benchmarking single mode: %s", args.mode)

    try:
        report = run_benchmark(
            image_path=image_path,
            modes=modes,
            runs=args.runs,
            warmup_runs=args.warmup,
            output_path=args.output,
        )
    finally:
        # Always clean up model from memory after benchmarking
        logger.info("Unloading VLMEngine after benchmark...")
        VLMLifecycleManager.unload()

    sys.exit(0 if report["summary"]["modes_succeeded"] > 0 else 1)


if __name__ == "__main__":
    main()
