"""Phase 3 Advanced Recognition Engine Command-Line Interface (CLI)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.phase3.config import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
)
from src.phase3.engine import AdvancedRecognitionEngine


def format_cli_output(result: Dict[str, Any]) -> str:
    """Format unified recognition result into a clean terminal report."""
    if not result.get("success", False):
        return f"\n[!] RECOGNITION ERROR: {result.get('error', 'Unknown error')} (Code: {result.get('error_code', 'ERROR')})"

    img = result.get("image", {})
    mode = result.get("mode", "all").upper()
    meta = result.get("metadata", {})

    lines = [
        "",
        "=" * 60,
        f"        ADVANCED IMAGE RECOGNITION REPORT ({mode})",
        "=" * 60,
        f"Image File:   {img.get('file_name')} ({img.get('width')}x{img.get('height')} {img.get('format')})",
        f"Total Latency: {meta.get('total_inference_ms')} ms",
        "-" * 60,
    ]

    # Classification Block
    if "classification" in result:
        clf = result["classification"]
        top1 = clf.get("top1", {})
        top5 = clf.get("top5", [])
        status = clf.get("status", "unknown")

        lines.extend([
            "",
            "1. IMAGE CLASSIFICATION (ImageNet 1,000 Classes):",
            f"   Model:        {clf.get('model')} ({clf.get('inference_ms')} ms)",
            f"   Primary:      {top1.get('label')} ({top1.get('confidence_percent'):.2f}%) [{status.upper()}]",
            "",
            "   Top 5 Ranked Predictions:",
        ])
        for i, item in enumerate(top5, 1):
            label_str = item.get("label", "").ljust(25)
            lines.append(f"     {i}. {label_str} {item.get('confidence_percent', 0.0):>6.2f}%")

        if not clf.get("is_confident", True):
            lines.extend([
                "",
                "   [!] UNCERTAINTY NOTICE:",
                f"       Confidence ({top1.get('confidence_percent'):.2f}%) is below {clf.get('confidence_threshold')*100:.0f}% threshold.",
            ])

    # Detection Block
    if "detection" in result:
        det = result["detection"]
        count = det.get("count", 0)
        objects = det.get("objects", [])

        lines.extend([
            "",
            "-" * 60,
            "2. OBJECT DETECTION (COCO 90 Object Classes):",
            f"   Model:        {det.get('model')} ({det.get('inference_ms')} ms)",
            f"   Detected:     {count} object(s)",
            "",
        ])

        if count == 0:
            lines.append("   (No objects detected above the confidence threshold)")
        else:
            for i, obj in enumerate(objects, 1):
                box = obj["box"]
                lines.append(
                    f"   [{i}] {obj.get('label').upper()} ({obj.get('confidence_percent'):.1f}%) "
                    f"-> Box: [x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}] "
                    f"({box['width']}x{box['height']} px)"
                )

        if "rendered_image_path" in det:
            lines.extend([
                "",
                f"   Rendered Visualization: {det['rendered_image_path']}",
            ])

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def main() -> None:
    """CLI Entrypoint for Phase 3 Advanced Recognition."""
    parser = argparse.ArgumentParser(
        description="Phase 3 Advanced Image Recognition & Object Detection Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image (JPG, PNG, WEBP, BMP)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "classification", "detection"],
        default="all",
        help="Inference mode to execute",
    )
    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        help="Confidence cutoff for classification certainty",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        help="Confidence cutoff for object detection boxes",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of classification predictions to return",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable rendering bounding boxes to disk",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report",
    )

    args = parser.parse_args()

    engine = AdvancedRecognitionEngine()
    result = engine.recognize(
        args.image,
        mode=args.mode,
        classification_threshold=args.classification_threshold,
        detection_threshold=args.detection_threshold,
        render_visualization=not args.no_render,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_cli_output(result))


if __name__ == "__main__":
    main()
