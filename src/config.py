"""Central configuration management for CIFAR-10 image recognition."""

import os
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np

# ==============================================================================
# BASE PATHS
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

MODEL_SAVE_PATH: Path = MODELS_DIR / "image_classifier.keras"
METADATA_SAVE_PATH: Path = MODELS_DIR / "metadata.json"
TRAINING_HISTORY_PLOT_PATH: Path = PLOTS_DIR / "training_history.png"
CONFUSION_MATRIX_PLOT_PATH: Path = PLOTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH: Path = METRICS_DIR / "classification_report.json"
TRAINING_METRICS_PATH: Path = METRICS_DIR / "training_history.json"

# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================
IMAGE_HEIGHT: int = 32
IMAGE_WIDTH: int = 32
CHANNELS: int = 3
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
NUM_CLASSES: int = 10

CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Validation split fraction from the 50,000 CIFAR-10 training pool
VAL_SPLIT_RATIO: float = 0.2  # 40,000 train, 10,000 val

# ==============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED: int = 42
BATCH_SIZE: int = 64
EPOCHS: int = 30
INITIAL_LEARNING_RATE: float = 1e-3
MIN_LEARNING_RATE: float = 1e-6

# Callback parameters
EARLY_STOPPING_PATIENCE: int = 7
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5

# Augmentation parameters
AUGMENTATION_CONFIG = {
    "random_flip": "horizontal",
    "random_rotation_factor": 0.08,     # +/- ~15 degrees
    "random_translation_factor": 0.08,  # +/- 8% height & width shift
    "random_zoom_factor": 0.08,         # +/- 8% zoom
}


def ensure_directories() -> None:
    """Ensure all required output and model directories exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
