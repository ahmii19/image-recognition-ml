"""Out-of-Distribution (OOD) Analysis, Softmax Limitation Investigation, and Recognition Capability Studies."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from src.phase3.config import PHASE3_METRICS_DIR, ensure_phase3_directories
from src.phase3.engine import AdvancedRecognitionEngine


def create_ood_test_artifacts(output_dir: Path) -> List[Tuple[str, str, Path]]:
    """Create controlled out-of-distribution synthetic test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    # 1. Pure Uniform Color Patch (No Semantic Object)
    p1 = output_dir / "ood_uniform_blue.png"
    img1 = Image.new("RGB", (400, 400), color=(30, 80, 200))
    img1.save(p1)
    samples.append(("uniform_blue", "Abstract uniform color field (no object)", p1))

    # 2. Random White Noise (Pure Random Entropy)
    p2 = output_dir / "ood_gaussian_noise.png"
    noise = np.random.randint(0, 256, size=(400, 400, 3), dtype=np.uint8)
    img2 = Image.fromarray(noise, mode="RGB")
    img2.save(p2)
    samples.append(("gaussian_noise", "Random pixel noise (no geometry)", p2))

    # 3. Geometric Shapes (Abstract Concentric Rings)
    p3 = output_dir / "ood_abstract_geometry.png"
    img3 = Image.new("RGB", (400, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img3)
    for r in range(20, 180, 25):
        draw.ellipse([200 - r, 200 - r, 200 + r, 200 + r], outline=(200, 50, 80), width=4)
    img3.save(p3)
    samples.append(("abstract_geometry", "Mathematical concentric rings", p3))

    # 4. Text-heavy Document / Code snippet
    p4 = output_dir / "ood_text_pattern.png"
    img4 = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img4)
    for y in range(20, 380, 20):
        draw.line([30, y, 370, y], fill=(60, 60, 60), width=3)
    img4.save(p4)
    samples.append(("striped_text_pattern", "Parallel horizontal stripe pattern", p4))

    return samples


def run_ood_investigation() -> Dict[str, Any]:
    """Execute the Out-of-Distribution and Softmax Reality Investigation.

    Returns:
        Structured findings dictionary.
    """
    from typing import Tuple
    ensure_phase3_directories()
    scratch_ood_dir = Path("scratch") / "ood_samples"
    ood_samples = create_ood_test_artifacts(scratch_ood_dir)

    engine = AdvancedRecognitionEngine(lazy_load=False)

    findings: Dict[str, Any] = {
        "title": "Phase 3 Out-of-Distribution & Softmax Normalization Analysis",
        "description": (
            "Empirical demonstration of why closed-set Softmax classification and bounding box detectors "
            "cannot automatically identify out-of-vocabulary inputs as 'unknown' without calibrated density estimation."
        ),
        "experiments": [],
        "key_takeaways": [
            "Softmax normalization forces probabilities to sum to 1.0, often assigning arbitrary labels to noise.",
            "Confidence thresholds filter weak detections, but high confidence can still occur on out-of-distribution textures.",
            "Closed-set recognition (ImageNet 1,000 / COCO 90) requires explicit open-vocabulary grounding (e.g. CLIP/OWL-ViT) for arbitrary vocabulary recognition."
        ],
    }

    for name, description, img_path in ood_samples:
        rec_res = engine.recognize(img_path, mode="all", render_visualization=False)
        clf = rec_res.get("classification", {})
        det = rec_res.get("detection", {})

        top1 = clf.get("top1", {})
        det_count = det.get("count", 0)

        findings["experiments"].append({
            "sample_name": name,
            "description": description,
            "classification_output": {
                "top1_label": top1.get("label"),
                "confidence_percent": top1.get("confidence_percent"),
                "status": clf.get("status"),
            },
            "detection_output": {
                "status": det.get("status"),
                "objects_detected_count": det_count,
            },
            "analysis": (
                f"Classifier assigned '{top1.get('label')}' with {top1.get('confidence_percent')}% confidence. "
                f"Detector returned {det_count} objects above threshold."
            ),
        })

    out_file = PHASE3_METRICS_DIR / "ood_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print(f"[INFO] OOD Experiment findings exported to: {out_file}")
    return findings


if __name__ == "__main__":
    from typing import Tuple
    run_ood_investigation()
