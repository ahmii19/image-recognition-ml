"""Inference CLI and prediction pipeline for single-image classification."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    METADATA_SAVE_PATH,
    MODEL_SAVE_PATH,
    PREDICTIONS_DIR,
    ensure_directories,
)
from src.preprocessing import preprocess_single_image


class ImageClassifier:
    """Production inference wrapper for trained CIFAR-10 CNN model."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_SAVE_PATH,
        metadata_path: Union[str, Path] = METADATA_SAVE_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{self.model_path}'. "
                "Please train the model first by executing: python -m src.train"
            )

        print(f"[INFO] Loading trained model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        # Load class names from metadata if available, otherwise fallback to config
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", CLASS_NAMES)
        else:
            self.class_names = CLASS_NAMES

    def predict(
        self,
        image_input: Union[str, Path],
        top_k: int = 3,
    ) -> Dict[str, Union[str, float, List[Dict[str, Union[str, float]]]]]:
        """Run image recognition inference on a single image.

        Args:
            image_input: Path to the image file.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary containing prediction label, confidence, and top-k rankings.
        """
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Target image not found at: {image_path}")

        # Preprocess image to shape (1, 32, 32, 3), float32 in [0, 1]
        preprocessed_img = preprocess_single_image(str(image_path))

        # Model forward pass -> shape (1, 10)
        probabilities = self.model.predict(preprocessed_img, verbose=0)[0]

        # Extract top-k sorted indices in descending order
        top_k = min(top_k, len(self.class_names))
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        top_predictions = []
        for idx in top_indices:
            top_predictions.append(
                {
                    "class": self.class_names[idx],
                    "class_index": int(idx),
                    "confidence_percent": float(probabilities[idx] * 100.0),
                    "probability": float(probabilities[idx]),
                }
            )

        primary_prediction = top_predictions[0]

        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "prediction": primary_prediction["class"],
            "confidence_percent": primary_prediction["confidence_percent"],
            "top_k": top_predictions,
            "all_probabilities": {
                name: float(prob) for name, prob in zip(self.class_names, probabilities)
            },
        }

        return result


def format_prediction_output(result: Dict, top_k: int = 3) -> str:
    """Format prediction dictionary into the standardized user-facing terminal layout."""
    lines = [
        "=" * 40,
        "IMAGE RECOGNITION",
        "=" * 40,
        "",
        f"Image: {result['image_name']}",
        "",
        "Prediction:",
        f"{result['prediction']}",
        "",
        "Confidence:",
        f"{result['confidence_percent']:.2f}%",
        "",
        f"Top {top_k}:",
    ]

    for rank, item in enumerate(result["top_k"], start=1):
        class_str = item["class"].ljust(12)
        lines.append(f"{rank}. {class_str} {item['confidence_percent']:>6.2f}%")

    lines.extend([
        "",
        "=" * 40,
        "[NOTE] CIFAR-10 Scope:",
        "This model is trained exclusively on 10 CIFAR-10 classes:",
        f"({', '.join(CLASS_NAMES)})",
        "It cannot recognize arbitrary objects outside this predefined domain.",
        "=" * 40,
    ])

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running single-image inference."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Image Recognition Inference CLI"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to target image file (JPEG/PNG)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_SAVE_PATH),
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top candidates to display (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )

    args = parser.parse_args()

    try:
        classifier = ImageClassifier(model_path=args.model_path)
        result = classifier.predict(args.image_path, top_k=args.top_k)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_prediction_output(result, top_k=args.top_k))

    except Exception as err:
        print(f"\n[ERROR] Inference failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
