"""Central configuration management for Phase 2 Transfer Learning."""

from pathlib import Path
from typing import Dict, List, Tuple
from src.config import BASE_DIR, set_reproducible_seed

# ==============================================================================
# PHASE 2 PATHS
# ==============================================================================
PHASE2_DATA_DIR: Path = BASE_DIR / "data" / "phase2"
PHASE2_MODELS_DIR: Path = BASE_DIR / "models" / "phase2"
PHASE2_OUTPUTS_DIR: Path = BASE_DIR / "outputs" / "phase2"
PHASE2_PLOTS_DIR: Path = PHASE2_OUTPUTS_DIR / "plots"
PHASE2_METRICS_DIR: Path = PHASE2_OUTPUTS_DIR / "metrics"
PHASE2_PREDICTIONS_DIR: Path = PHASE2_OUTPUTS_DIR / "predictions"

MOBILENET_MODEL_PATH: Path = PHASE2_MODELS_DIR / "mobilenetv2_best.keras"
EFFICIENTNET_MODEL_PATH: Path = PHASE2_MODELS_DIR / "efficientnetb0_best.keras"
SELECTED_MODEL_PATH: Path = PHASE2_MODELS_DIR / "selected_model.keras"
PHASE2_METADATA_PATH: Path = PHASE2_MODELS_DIR / "metadata.json"

PHASE2_TRAINING_PLOT_PATH: Path = PHASE2_PLOTS_DIR / "training_comparison.png"
PHASE2_CONFUSION_MATRIX_PATH: Path = PHASE2_PLOTS_DIR / "confusion_matrix.png"
PHASE2_METRICS_PATH: Path = PHASE2_METRICS_DIR / "comparison_metrics.json"
REAL_WORLD_PREDICTIONS_PATH: Path = PHASE2_OUTPUTS_DIR / "real_world_predictions.json"

# ==============================================================================
# DATASET PARAMETERS (High-Resolution Real-World Image Classification)
# ==============================================================================
DATASET_NAME: str = "flower_photos"
DATASET_URL: str = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATASET_LICENSE: str = "Creative Commons / Public Domain (TensorFlow Official Datasets)"

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
CHANNELS: int = 3
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

CLASS_NAMES: List[str] = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
]
NUM_CLASSES: int = len(CLASS_NAMES)

TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
RANDOM_SEED: int = 42

# ==============================================================================
# TRANSFER LEARNING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE: int = 32

# Stage A: Feature Extraction (Backbone frozen)
FEATURE_EXTRACT_EPOCHS: int = 5
FEATURE_EXTRACT_LR: float = 1e-3

# Stage B: Fine-Tuning (Top backbone blocks unfrozen)
FINE_TUNE_EPOCHS: int = 5
FINE_TUNE_LR: float = 1e-5

# Layer unfreezing thresholds
# MobileNetV2 has ~154 layers total; unfreezing from layer 100 onwards fine-tunes top bottleneck blocks
MOBILENET_FINE_TUNE_AT: int = 100
# EfficientNetB0 has ~238 layers total; unfreezing from layer 180 onwards fine-tunes top MBConv blocks
EFFICIENTNET_FINE_TUNE_AT: int = 180

# Callbacks
EARLY_STOPPING_PATIENCE: int = 4
REDUCE_LR_PATIENCE: int = 2
REDUCE_LR_FACTOR: float = 0.5


def ensure_phase2_directories() -> None:
    """Ensure all required Phase 2 output and model directories exist."""
    directories = [
        PHASE2_DATA_DIR,
        PHASE2_MODELS_DIR,
        PHASE2_OUTPUTS_DIR,
        PHASE2_PLOTS_DIR,
        PHASE2_METRICS_DIR,
        PHASE2_PREDICTIONS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
