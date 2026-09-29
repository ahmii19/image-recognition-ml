"""Data preprocessing, augmentation pipelines, and input transformation utilities."""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

from src.config import (
    AUGMENTATION_CONFIG,
    BATCH_SIZE,
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize image pixel intensities from [0, 255] integer range to [0.0, 1.0] float32.

    Why Normalization is Required:
    -----------------------------
    1. Numerical Stability: Raw 8-bit integer pixels (0 to 255) cause large activation
       magnitudes in early convolutional layers, leading to unstable or exploding gradients.
    2. Faster Convergence: When features are bounded in [0, 1], loss surfaces are smoother,
       allowing gradient descent (e.g., Adam) to take uniform step sizes in parameter space.
    3. Uniform Scaling: Ensures all color channels and spatial locations contribute on an
       equal scale without individual features dominating the forward/backward passes.

    Args:
        images: NumPy array of shape (N, H, W, C) with values in range [0, 255].

    Returns:
        NumPy array of shape (N, H, W, C) with dtype float32 and values in [0.0, 1.0].
    """
    if images.dtype != np.float32:
        return (images / 255.0).astype(np.float32)
    return images


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """Build a Keras Sequential layer for data augmentation during training.

    Why these augmentations are selected:
    ------------------------------------
    - RandomFlip("horizontal"): A dog or car flipped horizontally is still a valid dog or car.
      (Vertical flip is avoided because upside-down cars or horses are unnatural in CIFAR-10).
    - RandomRotation(0.08): Subtle rotation (+/- ~15 degrees) makes the CNN invariant to
      minor camera tilts without clipping critical features on low-res (32x32) canvases.
    - RandomTranslation(0.08, 0.08): Shifting +/- 8% teaches spatial translation invariance.
    - RandomZoom(0.08): Simulates varying distances from the camera sensor.

    Returns:
        tf.keras.Sequential container of augmentation layers.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                AUGMENTATION_CONFIG["random_flip"],
                seed=RANDOM_SEED,
                name="aug_random_flip",
            ),
            layers.RandomRotation(
                AUGMENTATION_CONFIG["random_rotation_factor"],
                fill_mode="nearest",
                seed=RANDOM_SEED,
                name="aug_random_rotation",
            ),
            layers.RandomTranslation(
                height_factor=AUGMENTATION_CONFIG["random_translation_factor"],
                width_factor=AUGMENTATION_CONFIG["random_translation_factor"],
                fill_mode="nearest",
                seed=RANDOM_SEED,
                name="aug_random_translation",
            ),
            layers.RandomZoom(
                height_factor=AUGMENTATION_CONFIG["random_zoom_factor"],
                width_factor=AUGMENTATION_CONFIG["random_zoom_factor"],
                fill_mode="nearest",
                seed=RANDOM_SEED,
                name="aug_random_zoom",
            ),
        ],
        name="data_augmentation",
    )


def create_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
    is_training: bool = False,
    augment: bool = False,
    shuffle_buffer: Optional[int] = None,
) -> tf.data.Dataset:
    """Create an optimized tf.data.Dataset pipeline with streaming normalization, shuffling, and prefetching.

    Args:
        images: uint8 [0..255] or float32 [0..1] array of images (N, 32, 32, 3).
        labels: Integer label array (N,).
        batch_size: Number of samples per batch.
        is_training: Whether the dataset is for training (enables shuffling).
        augment: Whether to apply the data augmentation pipeline.
        shuffle_buffer: Number of items for the shuffle buffer.

    Returns:
        Configured tf.data.Dataset yielding (batch_images, batch_labels).
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        buffer_size = shuffle_buffer if shuffle_buffer is not None else min(10000, len(images))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=RANDOM_SEED)

    dataset = dataset.batch(batch_size)

    # Normalize on the fly per mini-batch to save RAM
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0 if x.dtype != tf.float32 else x, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if is_training and augment:
        aug_layer = get_data_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch asynchronously to eliminate CPU-GPU I/O bottlenecks
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocess_single_image(
    image_input: Union[str, Image.Image],
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> np.ndarray:
    """Load, convert to RGB, resize, and normalize a single input image for model inference.

    Args:
        image_input: Filepath string or PIL Image object.
        target_size: Target (height, width) tuple, defaults to (32, 32).

    Returns:
        Batched normalized float32 NumPy array of shape (1, 32, 32, 3) with values in [0, 1].

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file is not a readable image format.
    """
    if isinstance(image_input, str):
        try:
            pil_img = Image.open(image_input)
        except Exception as err:
            raise ValueError(f"Failed to open image file '{image_input}': {err}") from err
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError(f"Expected file path string or PIL.Image, got {type(image_input)}")

    # Ensure RGB (3 channels) even if source is RGBA or Grayscale
    pil_img = pil_img.convert("RGB")

    # Resize using high-quality Lanczos resampling
    pil_img = pil_img.resize(target_size, resample=Image.Resampling.LANCZOS)

    # Convert to float32 NumPy array and scale to [0.0, 1.0]
    img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

    # Expand dims to create batch dimension: (1, 32, 32, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array
