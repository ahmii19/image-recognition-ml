"""Dataset downloading, stratified partitioning, augmentation, and input transformation for Phase 2."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet, mobilenet_v2

from src.phase2_config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_NAME,
    DATASET_URL,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PHASE2_DATA_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    ensure_phase2_directories,
)


def download_and_prepare_dataset() -> Path:
    """Download and extract the high-resolution real-world dataset.

    Dataset Details:
    ----------------
    - Name: flower_photos
    - Source: TensorFlow Official Datasets (Google Cloud Storage)
    - Classes (5): daisy, dandelion, roses, sunflowers, tulips
    - Images: ~3,670 variable-resolution real-world photos
    - Format: RGB JPEGs of varying sizes and aspect ratios

    Returns:
        Path to the extracted dataset directory.
    """
    ensure_phase2_directories()
    cache_dir = PHASE2_DATA_DIR / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Preparing Phase 2 dataset from {DATASET_URL}...")
    dataset_dir = tf.keras.utils.get_file(
        fname=DATASET_NAME,
        origin=DATASET_URL,
        untar=True,
        cache_dir=str(PHASE2_DATA_DIR),
        cache_subdir="dataset_cache",
    )
    extracted_path = Path(dataset_dir)
    print(f"[INFO] Dataset ready at: {extracted_path}")
    return extracted_path


def get_image_file_paths_and_labels(dataset_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """Scan dataset directory, discover valid RGB image files, and map class labels.

    Args:
        dataset_dir: Path to directory containing class subfolders.

    Returns:
        Tuple of (file_paths, integer_labels, class_names).
    """
    # If extraction created a nested subfolder (e.g. flower_photos/flower_photos)
    if (dataset_dir / dataset_dir.name).is_dir():
        candidate = dataset_dir / dataset_dir.name
        if any(d.is_dir() for d in candidate.iterdir()):
            dataset_dir = candidate

    file_paths: List[str] = []
    labels: List[int] = []

    # Filter out non-directory entries (like LICENSE.txt)
    discovered_classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(discovered_classes)}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for class_name in discovered_classes:
        class_folder = dataset_dir / class_name
        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                file_paths.append(str(img_file))
                labels.append(class_to_idx[class_name])

    return file_paths, labels, discovered_classes


def create_phase2_splits(
    file_paths: List[str],
    labels: List[int],
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Perform a reproducible stratified train/validation/test split on dataset paths.

    Prevents data leakage by partitioning filepaths before any image loading or augmentation.

    Args:
        file_paths: List of image filepaths.
        labels: List of integer class labels.
        train_ratio: Proportion for training (e.g. 0.70).
        val_ratio: Proportion for validation (e.g. 0.15).
        test_ratio: Proportion for testing (e.g. 0.15).
        random_seed: Random state seed.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to (paths, labels) tuples.
    """
    # First split off the test set (15%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    # Next split remaining 85% into train (70% overall) and val (15% overall)
    # val_ratio relative to train_val is 0.15 / 0.85 ~ 0.1765
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_relative_ratio,
        random_state=random_seed,
        stratify=train_val_labels,
    )

    print(
        f"[INFO] Phase 2 Dataset Partitioning:\n"
        f"       - Training:   {len(train_paths):,} images ({len(train_paths)/len(file_paths)*100:.1f}%)\n"
        f"       - Validation: {len(val_paths):,} images ({len(val_paths)/len(file_paths)*100:.1f}%)\n"
        f"       - Test:       {len(test_paths):,} images ({len(test_paths)/len(file_paths)*100:.1f}%)"
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def get_phase2_augmentation_layer() -> tf.keras.Sequential:
    """Build a robust data augmentation layer for real-world image training.

    Techniques:
    - RandomFlip('horizontal'): Captures invariant horizontal perspectives.
    - RandomRotation(0.1): Captures minor photographic tilt (+/- 18 degrees).
    - RandomTranslation(0.08, 0.08): Handles framing and framing offsets.
    - RandomZoom(0.1): Accommodates varying camera-to-subject distances.
    - RandomContrast(0.1): Simulates varying ambient lighting conditions.

    Returns:
        tf.keras.Sequential container of augmentation layers.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=RANDOM_SEED, name="p2_aug_flip"),
            layers.RandomRotation(0.1, fill_mode="nearest", seed=RANDOM_SEED, name="p2_aug_rotation"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=RANDOM_SEED, name="p2_aug_translation"),
            layers.RandomZoom(0.1, fill_mode="nearest", seed=RANDOM_SEED, name="p2_aug_zoom"),
            layers.RandomContrast(0.1, seed=RANDOM_SEED, name="p2_aug_contrast"),
        ],
        name="phase2_data_augmentation",
    )


def load_and_preprocess_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    model_family: str = "mobilenet_v2",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load an image file from disk, decode JPEG/PNG, resize to 224x224, and apply backbone preprocessing.

    Args:
        file_path: Path string tensor.
        label: Label integer tensor.
        model_family: 'mobilenet_v2' or 'efficientnet_b0'.

    Returns:
        Tuple of (preprocessed_image_tensor, label_tensor).
    """
    img_bytes = tf.io.read_file(file_path)
    # Expand to 3 RGB channels (handles Grayscale or RGBA gracefully)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)

    if model_family.lower() == "mobilenet_v2":
        # Scales to [-1.0, 1.0]
        img = mobilenet_v2.preprocess_input(img)
    elif "efficientnet" in model_family.lower():
        # Scales using EfficientNet's internal normalizer
        img = efficientnet.preprocess_input(img)
    else:
        # Default scaling to [0.0, 1.0]
        img = img / 255.0

    return img, label


def create_phase2_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    model_family: str = "mobilenet_v2",
    batch_size: int = BATCH_SIZE,
    is_training: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """Build a high-throughput tf.data.Dataset pipeline with streaming I/O and prefetching.

    Args:
        file_paths: List of file path strings.
        labels: List of integer class labels.
        model_family: Model family name for appropriate normalization.
        batch_size: Mini-batch size.
        is_training: If True, shuffles paths.
        augment: If True, applies data augmentation.

    Returns:
        Configured tf.data.Dataset yielding (batch_images, batch_labels).
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=RANDOM_SEED)

    # Parallel file reading and decoding
    dataset = dataset.map(
        lambda p, l: load_and_preprocess_image(p, l, model_family=model_family),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)

    if is_training and augment:
        aug_layer = get_phase2_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocess_phase2_image(
    image_input: Union[str, Path, Image.Image],
    model_family: str = "mobilenet_v2",
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> np.ndarray:
    """Preprocess a single arbitrary real-world image for Phase 2 inference.

    Preserves aspect ratio via smart cropping / resizing before model-specific normalization.

    Args:
        image_input: Filepath string, Path, or PIL.Image.
        model_family: 'mobilenet_v2' or 'efficientnet_b0' or 'imagenet'.
        target_size: (height, width) tuple (224, 224).

    Returns:
        NumPy array of shape (1, 224, 224, 3) ready for model forward pass.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at '{img_path}'")
        pil_img = Image.open(str(img_path))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError(f"Expected file path or PIL.Image, got {type(image_input)}")

    # Ensure 3-channel RGB
    pil_img = pil_img.convert("RGB")

    # Aspect-ratio preserving resize & center crop
    width, height = pil_img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    pil_img = pil_img.crop((left, top, right, bottom))
    pil_img = pil_img.resize(target_size, resample=Image.Resampling.LANCZOS)

    img_array = np.asarray(pil_img, dtype=np.float32)

    # Model specific normalization
    if model_family.lower() == "mobilenet_v2":
        img_array = mobilenet_v2.preprocess_input(img_array)
    elif "efficientnet" in model_family.lower():
        img_array = efficientnet.preprocess_input(img_array)
    else:
        # Default mobilenet / imagenet standard [-1, 1]
        img_array = mobilenet_v2.preprocess_input(img_array)

    return np.expand_dims(img_array, axis=0)
