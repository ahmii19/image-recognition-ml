"""Dataset loading and train/validation/test partitioning for CIFAR-10."""

from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    VAL_SPLIT_RATIO,
    ensure_directories,
)


@dataclass
class DatasetSplits:
    """Container for raw partitioned CIFAR-10 datasets."""
    x_train: np.ndarray  # Shape: (40000, 32, 32, 3), dtype: uint8 [0..255]
    y_train: np.ndarray  # Shape: (40000,), dtype: int
    x_val: np.ndarray    # Shape: (10000, 32, 32, 3), dtype: uint8 [0..255]
    y_val: np.ndarray    # Shape: (10000,), dtype: int
    x_test: np.ndarray   # Shape: (10000, 32, 32, 3), dtype: uint8 [0..255]
    y_test: np.ndarray   # Shape: (10000,), dtype: int


def load_cifar10_dataset(
    val_split: float = VAL_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED,
) -> DatasetSplits:
    """Load the official CIFAR-10 dataset and split into train, validation, and test sets.

    Partitioning:
      - CIFAR-10 contains 60,000 32x32 color images across 10 classes.
      - 50,000 raw training images are split into:
          * 40,000 Training samples (80%)
          * 10,000 Validation samples (20%) using stratified sampling.
      - 10,000 Official Test samples remain untouched for final evaluation.

    Args:
        val_split: Proportion of training data to use for validation.
        random_seed: Seed for reproducible stratified split.

    Returns:
        DatasetSplits dataclass holding (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    import pathlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    local_cifar = pathlib.Path.home() / ".keras" / "datasets" / "cifar"
    processed_npz = PROCESSED_DATA_DIR / "cifar10_raw.npz"
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    if processed_npz.exists():
        print(f"[INFO] Loading CIFAR-10 from binary cache: {processed_npz}...")
        data = np.load(processed_npz)
        x_train_full, y_train_full = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    elif (local_cifar / "train").exists() and (local_cifar / "test").exists():
        print(f"[INFO] Loading CIFAR-10 from local image directory: {local_cifar}...")

        def _read_image(img_file: pathlib.Path):
            label_name = img_file.stem.split("_")[-1]
            if label_name in class_to_idx:
                with Image.open(img_file) as im:
                    return np.array(im.convert("RGB"), dtype=np.uint8), class_to_idx[label_name]
            return None

        def load_folder_parallel(folder_path: pathlib.Path):
            files = list(folder_path.glob("*.png"))
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(executor.map(_read_image, files))
            valid = [r for r in results if r is not None]
            images = np.array([r[0] for r in valid], dtype=np.uint8)
            labels = np.array([r[1] for r in valid], dtype=np.int32)
            return images, labels

        x_train_full, y_train_full = load_folder_parallel(local_cifar / "train")
        x_test, y_test = load_folder_parallel(local_cifar / "test")

        # Save to npz cache for future instant access
        ensure_directories()
        np.savez_compressed(processed_npz, x_train=x_train_full, y_train=y_train_full, x_test=x_test, y_test=y_test)
        print(f"[INFO] Binary cache saved to {processed_npz}")
    else:
        print("[INFO] Downloading / Loading CIFAR-10 dataset via tf.keras.datasets...")
        (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train_full = y_train_full.squeeze().astype(np.int32)
        y_test = y_test.squeeze().astype(np.int32)

    # Perform reproducible stratified split so class distributions match
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_split,
        random_state=random_seed,
        stratify=y_train_full,
    )

    print(
        f"[INFO] Dataset loaded successfully:\n"
        f"       - Training set:   {x_train.shape[0]:,} images, shape: {x_train.shape[1:]}\n"
        f"       - Validation set: {x_val.shape[0]:,} images, shape: {x_val.shape[1:]}\n"
        f"       - Test set:       {x_test.shape[0]:,} images, shape: {x_test.shape[1:]}"
    )

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def save_dataset_samples(
    images: np.ndarray,
    labels: np.ndarray,
    num_samples_per_class: int = 2,
    output_path: str = None,
) -> str:
    """Save a sample grid showing representative images from each CIFAR-10 class.

    Args:
        images: Image array (N, 32, 32, 3), values 0-255 or 0.0-1.0.
        labels: Label array (N,).
        num_samples_per_class: Number of samples per class to display.
        output_path: Filepath where the plot should be saved.

    Returns:
        Path to the saved sample image grid.
    """
    ensure_directories()
    if output_path is None:
        output_path = str(PLOTS_DIR / "dataset_samples.png")

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        num_samples_per_class,
        figsize=(num_samples_per_class * 2.2, len(CLASS_NAMES) * 1.5),
    )

    for class_idx, class_name in enumerate(CLASS_NAMES):
        match_indices = np.where(labels == class_idx)[0][:num_samples_per_class]
        for col_idx, match_idx in enumerate(match_indices):
            ax = axes[class_idx, col_idx] if len(CLASS_NAMES) > 1 else axes[col_idx]
            sample_img = images[match_idx]
            if sample_img.dtype == np.uint8:
                sample_img = sample_img.astype("float32") / 255.0
            ax.imshow(sample_img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(f"Class {class_idx}: {class_name}", fontsize=10, loc="left", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Dataset class samples saved to: {output_path}")
    return output_path
