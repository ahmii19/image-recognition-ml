"""Production-oriented Phase 2 Real-World Image Recognition Inference CLI."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

from src.phase2_config import (
    CLASS_NAMES,
    PHASE2_METADATA_PATH,
    SELECTED_MODEL_PATH,
    ensure_phase2_directories,
)
from src.phase2_data import preprocess_phase2_image


class Phase2ImageRecognizer:
    """Real-world image recognition engine supporting both fine-tuned domain models and ImageNet-1K general recognition."""

    def __init__(
        self,
        mode: str = "transfer_learned",
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.mode = mode.lower()

        if self.mode == "transfer_learned":
            self.model_path = Path(model_path) if model_path else SELECTED_MODEL_PATH
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Phase 2 model checkpoint not found at: {self.model_path}. "
                    "Please run 'python -m src.phase2_train' first."
                )
            print(f"[INFO] Loading fine-tuned Transfer Learning model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)

            if PHASE2_METADATA_PATH.exists():
                with open(PHASE2_METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.class_names = meta.get("dataset_classes", CLASS_NAMES)
            else:
                self.class_names = CLASS_NAMES

        elif self.mode == "imagenet":
            print("[INFO] Initializing ImageNet-1000 general real-world recognition model (MobileNetV2)...")
            self.model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
            self.class_names = None
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'transfer_learned' or 'imagenet'.")

    def predict(
        self,
        image_input: Union[str, Path],
        top_k: int = 5,
        uncertainty_threshold: float = 40.0,
    ) -> Dict[str, Any]:
        """Classify a single image and return Top-K predictions with confidence analysis."""
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Target image file not found at: {img_path}")

        # Preprocess input image: aspect-preserving center-crop to 224x224
        img_batch = preprocess_phase2_image(img_path, model_family="mobilenet_v2")

        # Forward pass
        probs = self.model.predict(img_batch, verbose=0)[0]

        top_predictions: List[Dict[str, Any]] = []

        if self.mode == "transfer_learned":
            top_k_clamped = min(top_k, len(self.class_names))
            top_indices = np.argsort(probs)[::-1][:top_k_clamped]

            for idx in top_indices:
                top_predictions.append(
                    {
                        "class": self.class_names[idx],
                        "confidence_percent": float(probs[idx] * 100.0),
                        "probability": float(probs[idx]),
                    }
                )
        else:
            # Decode ImageNet-1000 predictions
            decoded = decode_predictions(np.expand_dims(probs, axis=0), top=top_k)[0]
            for _, class_name, prob in decoded:
                formatted_name = class_name.replace("_", " ")
                top_predictions.append(
                    {
                        "class": formatted_name,
                        "confidence_percent": float(prob * 100.0),
                        "probability": float(prob),
                    }
                )

        primary_pred = top_predictions[0]
        is_uncertain = primary_pred["confidence_percent"] < uncertainty_threshold

        return {
            "image_name": img_path.name,
            "image_path": str(img_path),
            "mode": self.mode,
            "prediction": primary_pred["class"],
            "confidence_percent": primary_pred["confidence_percent"],
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": uncertainty_threshold,
            "top_k": top_predictions,
        }


def format_phase2_output(result: Dict[str, Any], top_k: int = 5) -> str:
    """Format inference result into standardized user-facing terminal layout."""
    lines = [
        "=" * 40,
        "REAL-WORLD IMAGE RECOGNITION",
        "=" * 40,
        "",
        f"Image: {result['image_name']}",
        f"Mode:  {result['mode'].upper()}",
        "",
        f"Top {top_k} predictions:",
        "",
    ]

    for rank, item in enumerate(result["top_k"][:top_k], start=1):
        class_str = item["class"].ljust(22)
        lines.append(f"{rank}. {class_str} {item['confidence_percent']:>6.2f}%")

    lines.append("")

    if result["is_uncertain"]:
        lines.extend([
            "[!] UNCERTAINTY WARNING:",
            f"The highest prediction confidence ({result['confidence_percent']:.2f}%) is below the {result['uncertainty_threshold']}% threshold.",
            "This image may depict an out-of-domain object or contain ambiguous visual features.",
            "",
        ])

    lines.extend([
        "=" * 40,
        "[NOTE] Softmax Confidence Notice:",
        "Softmax output represents normalized relative activation across candidate classes,",
        "which can be overconfident for out-of-distribution inputs.",
        "=" * 40,
    ])

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for Phase 2 image recognition."""
    parser = argparse.ArgumentParser(description="Phase 2 Real-World Image Recognition CLI")
    parser.add_argument("image_path", type=str, help="Path to target image file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transfer_learned", "imagenet"],
        default="transfer_learned",
        help="Inference mode ('transfer_learned' for custom fine-tuned domain, 'imagenet' for general 1K classes)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top candidates (default: 5)")
    parser.add_argument("--model_path", type=str, default=None, help="Custom .keras model checkpoint path")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    try:
        recognizer = Phase2ImageRecognizer(mode=args.mode, model_path=args.model_path)
        res = recognizer.predict(args.image_path, top_k=args.top_k)

        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(format_phase2_output(res, top_k=args.top_k))
    except Exception as err:
        print(f"\n[ERROR] Phase 2 inference failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
