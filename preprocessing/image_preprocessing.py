"""
Preprocessing pipeline for crop images used by both the stage classification
and disease detection models. Handles resizing, normalization, augmentation
configuration, and dataset directory splitting.

Two decode functions are provided:
- decode_and_resize_raw: Returns float32 in [0, 255] range. Use this for
  inference with models trained using Keras preprocess_input (EfficientNet,
  MobileNetV2, etc.) — the caller applies preprocess_input after this.
- decode_and_resize: Returns float32 in [0, 1] range (legacy, for any code
  that still expects /255 normalized images).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger

logger = get_logger(__name__)


def decode_and_resize_raw(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Decode raw image bytes into a float32 numpy array in [0, 255] range.

    This is the correct preprocessing for models trained with Keras
    preprocess_input functions (EfficientNet, MobileNetV2, etc.), which
    expect raw pixel values and handle normalization themselves.

    Args:
        image_bytes: raw bytes from HTTP upload or file read
        target_size: (width, height) tuple for resizing

    Returns:
        numpy array of shape (H, W, 3) with dtype float32, values in [0, 255]
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImageProcessingError("OpenCV (cv2) is required for image preprocessing.") from exc

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProcessingError("Could not decode image. Ensure the file is a valid JPG/PNG.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32")  # [0, 255] — DO NOT divide by 255 here
    return img


def decode_and_resize(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Decode raw image bytes into a normalized numpy array in [0, 1] range.

    LEGACY function — kept for backward compatibility with any code that
    expects 0-1 normalized images. For new inference code using Keras
    pretrained models, use decode_and_resize_raw instead.

    Args:
        image_bytes: raw bytes from HTTP upload or file read
        target_size: (width, height) tuple for resizing

    Returns:
        numpy array of shape (H, W, 3) with dtype float32, values in [0, 1]
    """
    img = decode_and_resize_raw(image_bytes, target_size)
    img = img / 255.0
    return img


def to_batch(img_array: np.ndarray) -> np.ndarray:
    """Add a batch dimension: (H, W, C) -> (1, H, W, C)."""
    return np.expand_dims(img_array, axis=0)


def build_augmentation_generator(rescale: float = 1.0 / 255.0):
    """
    Returns a configured Keras ImageDataGenerator for training-time augmentation.

    NOTE: For models using preprocess_input (EfficientNet, MobileNetV2), do NOT
    use this function. Instead create an ImageDataGenerator with
    preprocessing_function=preprocess_input directly in the training script.
    This function is kept for backward compatibility with any custom models
    that use simple rescale=1/255 normalization.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2,
    )


def split_dataset_dir(
    source_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Optional utility: physically split an `<source_dir>/<class>/*.jpg` layout
    into `<source_dir>_split/train/<class>` and `<source_dir>_split/val/<class>`.
    Most training scripts instead use validation_split inside
    ImageDataGenerator, so this is provided for cases where a hard split on
    disk is preferred (e.g. for use with image_dataset_from_directory).
    """
    rng = np.random.default_rng(seed)
    source = Path(source_dir)
    if not source.exists():
        raise ImageProcessingError(f"Source directory does not exist: {source_dir}")

    dest = Path(str(source) + "_split")
    train_dir = dest / "train"
    val_dir = dest / "val"

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ImageProcessingError(f"No class subdirectories found in {source_dir}")

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.*"))
        rng.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_dir / class_dir.name / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir / class_dir.name / img.name)

    logger.info(f"Dataset split complete. Output written to {dest}")
