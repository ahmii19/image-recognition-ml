"""Phase 3 Configuration, Paths, Hyperparameters, and COCO Class Mappings."""

from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE3_DATA_DIR = PROJECT_ROOT / "data" / "phase3"
PHASE3_MODELS_DIR = PROJECT_ROOT / "models" / "phase3"
PHASE3_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "phase3"
PHASE3_DETECTIONS_DIR = PHASE3_OUTPUTS_DIR / "detections"
PHASE3_METRICS_DIR = PHASE3_OUTPUTS_DIR / "metrics"
PHASE3_PLOTS_DIR = PHASE3_OUTPUTS_DIR / "plots"

# Detector Model URLs and Files
DETECTOR_MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
DETECTOR_MODEL_ARCHIVE = f"{DETECTOR_MODEL_NAME}.tar.gz"
DETECTOR_MODEL_URL = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{DETECTOR_MODEL_ARCHIVE}"
DETECTOR_LOCAL_PATH = PHASE3_MODELS_DIR / DETECTOR_MODEL_NAME / "saved_model"

# ---------------------------------------------------------------------------
# Default Operational Thresholds
# ---------------------------------------------------------------------------
DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.40  # 40% probability
DEFAULT_DETECTION_CONFIDENCE_THRESHOLD = 0.40       # 40% probability
DEFAULT_DETECTION_IOU_THRESHOLD = 0.50              # Non-Maximum Suppression IoU

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_DIMENSION = 8192
MIN_IMAGE_DIMENSION = 10

# ---------------------------------------------------------------------------
# COCO 2017 Dataset Label Map (90 Classes)
# ---------------------------------------------------------------------------
COCO_LABELS: Dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def ensure_phase3_directories() -> None:
    """Create all required Phase 3 folders."""
    for directory in [
        PHASE3_DATA_DIR,
        PHASE3_MODELS_DIR,
        PHASE3_OUTPUTS_DIR,
        PHASE3_DETECTIONS_DIR,
        PHASE3_METRICS_DIR,
        PHASE3_PLOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
