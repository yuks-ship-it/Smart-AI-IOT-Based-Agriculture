"""
Train the Crop Growth Stage Classification model.

Expects an image directory structured as:

    datasets/crop_images/<crop>/stage/<stage_name>/*.jpg

e.g.
    datasets/crop_images/rice/stage/seedling/*.jpg
    datasets/crop_images/rice/stage/tillering/*.jpg
    datasets/crop_images/rice/stage/flowering/*.jpg

Architecture: MobileNetV2 (lightweight, good for small datasets, mobile-deployable)
Strategy: Two-phase training with frozen BatchNorm, LR warmup, class weights

Usage:
    python training/train_stage_model.py
    python training/train_stage_model.py --epochs 30 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config_loader import ConfigLoader, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = "datasets/crop_images"


def collect_stage_directory(data_dir: Path, tmp_pool_dir: Path) -> None:
    """
    Pools images from datasets/crop_images/<crop>/stage/<stage>/*.jpg into
    a flat structure tmp_pool_dir/<stage>/<crop>_<filename>.
    Only collects stages that have actual images (skips empty folders).
    """
    if tmp_pool_dir.exists():
        shutil.rmtree(tmp_pool_dir)
    tmp_pool_dir.mkdir(parents=True, exist_ok=True)

    settings = ConfigLoader.load("app_settings.json")
    crops = settings["supported_crops"]

    found_any = False
    for crop in crops:
        stage_root = data_dir / crop / "stage"
        if not stage_root.exists():
            continue
        for stage_dir in stage_root.iterdir():
            if not stage_dir.is_dir():
                continue
            dst = tmp_pool_dir / stage_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in stage_dir.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    found_any = True
                    shutil.copy(img_path, dst / f"{crop}_{img_path.name}")

    # Remove empty class directories (stages with no images)
    for stage_dir in tmp_pool_dir.iterdir():
        if stage_dir.is_dir() and not any(stage_dir.iterdir()):
            stage_dir.rmdir()

    if not found_any:
        raise FileNotFoundError(
            f"No images found under {data_dir}/<crop>/stage/<stage>/. "
            f"Place training images there before running this script."
        )


def main():
    parser = argparse.ArgumentParser(description="Train growth stage classification model.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.callbacks import (
        Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    settings = ConfigLoader.load("app_settings.json")
    stage_cfg = settings["models"]["stage_classification"]
    img_size = tuple(stage_cfg["image_size"])

    data_dir = resolve_path(args.data_dir)
    pooled_dir = resolve_path("datasets/_stage_pool")
    collect_stage_directory(data_dir, pooled_dir)

    # MobileNetV2's own preprocessing (scales to [-1, 1] internally)
    # STRONGER augmentation for small dataset (only 468 images)
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30.0,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        pooled_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        pooled_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    labels_by_index = {v: k for k, v in class_indices.items()}

    logger.info(f"Found {num_classes} stage classes: {list(class_indices.keys())}")
    logger.info(f"Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

    # Compute class weights (critical — flowering has 4x fewer images)
    total_samples = sum(train_gen.classes.tolist().count(i) for i in range(num_classes))
    class_weights = {}
    for cls_idx in range(num_classes):
        cls_count = train_gen.classes.tolist().count(cls_idx)
        class_weights[cls_idx] = total_samples / (num_classes * cls_count) if cls_count > 0 else 1.0
    logger.info(f"Class weights: {class_weights}")

    # Build model with MobileNetV2 backbone
    base_model = MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    # === Paths ===
    model_path = resolve_path(stage_cfg["path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = resolve_path("models/checkpoints/stage")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # === Callbacks ===
    best_checkpoint = ModelCheckpoint(
        str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    epoch_checkpoint = ModelCheckpoint(
        str(checkpoints_dir / "epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras"),
        monitor="val_accuracy",
        save_best_only=False,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # ===== Phase 1: Train top layers only (base frozen) =====
    # Same label smoothing in both phases (avoids loss jump at transition)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    phase1_epochs = max(args.epochs // 3, 8)
    logger.info(f"Phase 1: Training top layers for {phase1_epochs} epochs...")

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        class_weight=class_weights,
        callbacks=[best_checkpoint, epoch_checkpoint, reduce_lr],
    )

    # ===== Phase 2: Unfreeze top layers, keep BatchNorm FROZEN =====
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    # CRITICAL: Keep BatchNorm frozen to prevent statistics shift
    for layer in base_model.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    frozen_bn = sum(1 for l in base_model.layers[-20:]
                    if isinstance(l, tf.keras.layers.BatchNormalization) and not l.trainable)
    logger.info(f"Phase 2: Unfroze top 20 layers, kept {frozen_bn} BatchNorm FROZEN")

    # LR warmup schedule (prevents disruption from optimizer reset)
    class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, target_lr=5e-5, warmup_epochs=2, steps_per_epoch=30):
            super().__init__()
            self.target_lr = target_lr
            self.warmup_steps = warmup_epochs * steps_per_epoch
            self.total_warmup = float(self.warmup_steps)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_factor = tf.minimum(step / self.total_warmup, 1.0)
            return self.target_lr * warmup_factor

        def get_config(self):
            return {"target_lr": self.target_lr, "warmup_steps": self.warmup_steps}

    warmup_lr = WarmupSchedule(
        target_lr=5e-5,
        warmup_epochs=2,
        steps_per_epoch=len(train_gen),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    phase2_epochs = args.epochs - phase1_epochs
    logger.info(f"Phase 2: Fine-tuning with warmup LR (0 -> 5e-5 over 2 epochs)...")

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_epochs,
        class_weight=class_weights,
        callbacks=[best_checkpoint, epoch_checkpoint, early_stop],
    )

    # === Merge histories ===
    full_history = {}
    for key in history_phase1.history:
        full_history[key] = history_phase1.history[key] + history_phase2.history.get(key, [])

    # Save training history
    history_path = resolve_path("models/stage_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Save labels
    labels_path = resolve_path(stage_cfg["labels_path"])
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_by_index, f, indent=2)

    logger.info(f"Saved stage model to {model_path}")
    logger.info(f"Saved stage labels to {labels_path}")
    logger.info(f"Epoch checkpoints saved to {checkpoints_dir}")

    # Auto-generate chart
    try:
        from training.visualize_training import plot_training_charts
        chart_path = str(resolve_path("models/stage_training_chart.png"))
        plot_training_charts(full_history, save_path=chart_path, model_name="Growth Stage")
        logger.info(f"Training chart saved to {chart_path}")
    except Exception as exc:
        logger.warning(f"Could not auto-generate chart: {exc}")

    # Register version
    from utils.model_versioning import register_version

    final_val_accuracy = history_phase2.history.get("val_accuracy", [None])[-1]
    version_id = register_version(
        model_name="stage_model",
        live_file_paths=[stage_cfg["path"], stage_cfg["labels_path"]],
        metrics={"final_val_accuracy": round(float(final_val_accuracy), 4) if final_val_accuracy else None},
        algorithm="MobileNetV2",
    )
    logger.info(f"Registered as version {version_id}.")
    logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}" if final_val_accuracy else "No val accuracy recorded.")

    # Print summary
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE - Growth Stage Model")
    print(f"{'='*65}")
    print(f"  Classes: {list(class_indices.keys())}")
    print(f"  Best model:       {model_path}")
    print(f"  Checkpoints:      {checkpoints_dir}")
    print(f"  History:          {history_path}")
    print(f"  Chart:            models/stage_training_chart.png")
    print(f"  Labels:           {labels_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
