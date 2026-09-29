"""Visual rendering utility for object detection bounding boxes and confidence tags."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.phase3.config import PHASE3_DETECTIONS_DIR, ensure_phase3_directories

# Distinct, high-contrast palette for visual object categories
PALETTE: List[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00A2FF", "#3445FF", "#5B00FF", "#A300FF",
    "#FF00F5", "#FF0080", "#E83E8C", "#6F42C1", "#FD7E14",
]


def _get_color_for_label(label: str) -> str:
    """Deterministically assign a color based on label string hash."""
    h = sum(ord(c) for c in label)
    return PALETTE[h % len(PALETTE)]


def render_detections(
    pil_image: Image.Image,
    detections: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    line_thickness: Optional[int] = None,
) -> Path:
    """Render bounding boxes, category badges, and confidence scores onto an image.

    Args:
        pil_image: Original RGB PIL image.
        detections: List of detection dictionaries containing 'box', 'label', 'confidence_percent'.
        save_path: Optional explicit output path. If None, saves to outputs/phase3/detections/.
        line_thickness: Bounding box line thickness. Auto-scaled based on image resolution if None.

    Returns:
        Path to the saved rendered image.
    """
    ensure_phase3_directories()
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Auto-scale box thickness and font size based on image diagonal
    diagonal = (w ** 2 + h ** 2) ** 0.5
    thickness = line_thickness if line_thickness is not None else max(2, int(diagonal / 300))
    font_size = max(11, int(diagonal / 45))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for item in detections:
        box = item["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = item.get("label", "object")
        conf_pct = item.get("confidence_percent", item.get("confidence", 0.0) * 100.0)
        tag_text = f"{label} {conf_pct:.1f}%"

        color_hex = _get_color_for_label(label)
        color_rgb = ImageColor.getrgb(color_hex)

        # Draw main rectangle with thickness
        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=color_rgb,
            )

        # Compute text bounding box
        try:
            bbox = draw.textbbox((x1, y1), tag_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(tag_text) * (font_size * 0.6)
            text_h = font_size

        pad = max(2, int(font_size * 0.2))
        badge_y1 = max(0, y1 - text_h - (pad * 2))
        badge_y2 = badge_y1 + text_h + (pad * 2)
        badge_x2 = min(w, x1 + text_w + (pad * 2))

        # Fill background badge
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color_rgb)

        # Text label (contrasting white text)
        draw.text(
            (x1 + pad, badge_y1 + pad),
            tag_text,
            fill=(255, 255, 255),
            font=font,
        )

    # Determine destination
    if save_path:
        out_path = Path(save_path)
    else:
        out_path = PHASE3_DETECTIONS_DIR / "detection_rendered.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, format="PNG")
    return out_path
