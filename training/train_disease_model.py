"""
Train the Disease Detection model.

Features:
- EfficientNetB0 backbone with proper preprocess_input
- Two-phase training (frozen base → fine-tune top layers)
- Checkpoint saved EVERY epoch + best model saved separately
- Training chart auto-saved after training
- Gradient monitoring at epochs 6-8 (prints gradient norms per layer)
- Full training history saved as JSON

Usage:
    python training/train_disease_model.py --data-dir "D:/RiceDisease/RiceDisease" --epochs 25 --batch-size 16
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

DEFAULT_DATA_DIR = "datasets/crop_images/disease"


def pool_disease_directory(data_dir: Path, tmp_pool_dir: Path) -> None:
    """
    Pools datasets/crop_images/disease/<crop>/<disease>/*.jpg into
    tmp_pool_dir/<disease>/<crop>_<filename>.
    """
    if tmp_pool_dir.exists():
        shutil.rmtree(tmp_pool_dir)
    tmp_pool_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Disease dataset directory not found: {data_dir}")

    found_any = False
    for crop_dir in data_dir.iterdir():
        if not crop_dir.is_dir():
            continue
        for disease_dir in crop_dir.iterdir():
            if not disease_dir.is_dir():
                continue
            dst = tmp_pool_dir / disease_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in disease_dir.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    found_any = True
                    shutil.copy(img_path, dst / f"{crop_dir.name}_{img_path.name}")

    if not found_any:
        raise FileNotFoundError(
            f"No images found under {data_dir}/<crop>/<disease_class>/. "
            f"Place training images there before running this script."
        )


def compute_gradients(model, val_gen, num_batches=3):
    """
    Compute gradient norms for all trainable layers using a few validation batches.
    Returns a dict of {layer_name: gradient_norm}.
    """
    import tensorflow as tf

    gradient_norms = {}
    trainable_vars = model.trainable_variables

    total_grads = [tf.zeros_like(v) for v in trainable_vars]
    count = 0

    for i, (x_batch, y_batch) in enumerate(val_gen):
        if i >= num_batches:
            break
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, trainable_vars)
        for j, g in enumerate(grads):
            if g is not None:
                total_grads[j] = total_grads[j] + tf.abs(g)
        count += 1

    # Compute average gradient norms
    for var, grad_sum in zip(trainable_vars, total_grads):
        avg_grad = grad_sum / count if count > 0 else grad_sum
        norm = float(tf.norm(avg_grad).numpy())
        gradient_norms[var.name] = norm

    return gradient_norms


def print_gradient_report(gradient_norms: dict, epoch: int):
    """Print a formatted gradient report for a given epoch."""
    print(f"\n{'='*75}")
    print(f"  GRADIENT REPORT - Epoch {epoch}")
    print(f"{'='*75}")
    print(f"  {'Layer':<50} {'Gradient Norm':<15} {'Status'}")
    print(f"  {'-'*50} {'-'*15} {'-'*10}")

    sorted_grads = sorted(gradient_norms.items(), key=lambda x: x[1], reverse=True)

    vanishing_count = 0
    exploding_count = 0
    healthy_count = 0

    for name, norm in sorted_grads[:30]:  # Show top 30 layers
        short_name = name[-48:] if len(name) > 48 else name
        if norm < 1e-7:
            status = "VANISHING"
            vanishing_count += 1
        elif norm > 10.0:
            status = "EXPLODING"
            exploding_count += 1
        else:
            status = "OK"
            healthy_count += 1
        print(f"  {short_name:<50} {norm:<15.8f} {status}")

    print(f"\n  Summary: {healthy_count} healthy | {vanishing_count} vanishing | {exploding_count} exploding")
    
    all_norms = list(gradient_norms.values())
    print(f"  Mean gradient norm: {np.mean(all_norms):.8f}")
    print(f"  Max gradient norm:  {np.max(all_norms):.8f}")
    print(f"  Min gradient norm:  {np.min(all_norms):.8f}")
    print(f"{'='*75}\n")

    return {
        "epoch": epoch,
        "mean_norm": float(np.mean(all_norms)),
        "max_norm": float(np.max(all_norms)),
        "min_norm": float(np.min(all_norms)),
        "vanishing_layers": vanishing_count,
        "exploding_layers": exploding_count,
        "healthy_layers": healthy_count,
        "top_10_layers": {k: v for k, v in sorted_grads[:10]},
    }


class GradientMonitorCallback:
    """Monitors gradients at specified epochs."""

    def __init__(self, model, val_gen, monitor_epochs, gradient_reports):
        self.model = model
        self.val_gen = val_gen
        self.monitor_epochs = monitor_epochs
        self.gradient_reports = gradient_reports
        self.current_epoch = 0

    def on_epoch_end(self, epoch):
        self.current_epoch = epoch + 1
        if self.current_epoch in self.monitor_epochs:
            logger.info(f"Computing gradients at epoch {self.current_epoch}...")
            grad_norms = compute_gradients(self.model, self.val_gen, num_batches=3)
            report = print_gradient_report(grad_norms, self.current_epoch)
            self.gradient_reports.append(report)


def main():
    parser = argparse.ArgumentParser(description="Train disease detection model.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--crop", type=str, default="rice",
                        help="Crop name (rice/wheat/maize). Determines model save path.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-epochs", type=str, default="6,7,8",
                        help="Comma-separated epochs to monitor gradients (default: 6,7,8)")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.callbacks import (
        Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Parse gradient monitoring epochs
    gradient_monitor_epochs = [int(e.strip()) for e in args.gradient_epochs.split(",")]

    settings = ConfigLoader.load("app_settings.json")
    disease_cfg = settings["models"]["disease_detection"]
    img_size = tuple(disease_cfg["image_size"])

    # Per-crop model paths
    crop = args.crop.strip().lower()
    crop_cfg = disease_cfg["per_crop"].get(crop)
    if not crop_cfg:
        logger.error(f"No config found for crop '{crop}' in app_settings.json. Add it to models.disease_detection.per_crop.")
        sys.exit(1)
    model_save_path = resolve_path(crop_cfg["path"])
    labels_save_path = resolve_path(crop_cfg["labels_path"])
    logger.info(f"Training disease model for crop: {crop}")

    data_dir = resolve_path(args.data_dir)
    pooled_dir = resolve_path(f"datasets/_disease_pool_{crop}")
    pool_disease_directory(data_dir, pooled_dir)

    # EfficientNet's own preprocessing
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
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

    logger.info(f"Found {num_classes} disease classes: {list(class_indices.keys())}")
    logger.info(f"Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

    # Build model with EfficientNetB0 backbone
    base_model = EfficientNetB0(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    # === Checkpoint paths ===
    model_path = model_save_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = resolve_path(f"models/checkpoints/disease_{crop}")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # === Callbacks ===
    # Save best model
    best_checkpoint = ModelCheckpoint(
        str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    # Save EVERY epoch checkpoint
    epoch_checkpoint = ModelCheckpoint(
        str(checkpoints_dir / "epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras"),
        monitor="val_accuracy",
        save_best_only=False,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Gradient monitor (manual — called between phases)
    gradient_reports = []

    # Custom callback for gradient monitoring during training
    class GradientCallback(Callback):
        def __init__(self, monitor_epochs, val_gen, gradient_reports, epoch_offset=0):
            super().__init__()
            self.monitor_epochs = monitor_epochs
            self.val_gen = val_gen
            self.gradient_reports = gradient_reports
            self.epoch_offset = epoch_offset

        def on_epoch_end(self, epoch, logs=None):
            actual_epoch = epoch + 1 + self.epoch_offset
            if actual_epoch in self.monitor_epochs:
                logger.info(f"Computing gradients at epoch {actual_epoch}...")
                grad_norms = compute_gradients(self.model, self.val_gen, num_batches=3)
                report = print_gradient_report(grad_norms, actual_epoch)
                self.gradient_reports.append(report)

    # ===== Phase 1: Train top layers only (base frozen) =====
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_epochs = max(args.epochs // 3, 5)
    logger.info(f"Phase 1: Training top layers for {phase1_epochs} epochs...")
    logger.info(f"Gradient monitoring scheduled at epochs: {gradient_monitor_epochs}")

    grad_cb_phase1 = GradientCallback(
        gradient_monitor_epochs, val_gen, gradient_reports, epoch_offset=0
    )

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=[best_checkpoint, epoch_checkpoint, reduce_lr, grad_cb_phase1],
    )

    # ===== Phase 2: Unfreeze top layers for fine-tuning =====
    # 20 layers unfrozen — sweet spot between capacity and overfitting control
    # LR 5e-5 — gentle enough to not disrupt pretrained features
    # Class weights — compensates for panicle_blast having fewer samples
    # Label smoothing — prevents overconfident predictions, stabilizes val loss
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Compute class weights to handle imbalanced data
    total_samples = sum(train_gen.classes.tolist().count(i) for i in range(num_classes))
    class_weights = {}
    for cls_idx in range(num_classes):
        cls_count = train_gen.classes.tolist().count(cls_idx)
        class_weights[cls_idx] = total_samples / (num_classes * cls_count) if cls_count > 0 else 1.0
    logger.info(f"Class weights: {class_weights}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    phase2_epochs = args.epochs - phase1_epochs
    logger.info(f"Phase 2: Fine-tuning top 30 layers for {phase2_epochs} epochs...")

    grad_cb_phase2 = GradientCallback(
        gradient_monitor_epochs, val_gen, gradient_reports, epoch_offset=phase1_epochs
    )

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_epochs,
        class_weight=class_weights,
        callbacks=[best_checkpoint, epoch_checkpoint, early_stop, grad_cb_phase2],

    )

    # === Merge both phase histories ===
    full_history = {}
    for key in history_phase1.history:
        full_history[key] = history_phase1.history[key] + history_phase2.history.get(key, [])

    # Save training history to JSON
    history_path = resolve_path(f"models/{crop}_disease_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Save gradient reports to JSON
    gradient_path = resolve_path(f"models/{crop}_disease_gradient_reports.json")
    with open(gradient_path, "w", encoding="utf-8") as f:
        json.dump(gradient_reports, f, indent=2, default=str)
    logger.info(f"Saved gradient reports to {gradient_path}")

    # Save labels
    labels_path = labels_save_path
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_by_index, f, indent=2)

    logger.info(f"Saved disease model to {model_path}")
    logger.info(f"Saved disease labels to {labels_path}")
    logger.info(f"Epoch checkpoints saved to {checkpoints_dir}")

    # === Auto-generate training chart ===
    try:
        from training.visualize_training import plot_training_charts
        chart_path = str(resolve_path(f"models/{crop}_disease_training_chart.png"))
        plot_training_charts(full_history, save_path=chart_path, model_name="Disease Detection")
        logger.info(f"Training chart saved to {chart_path}")
    except Exception as exc:
        logger.warning(f"Could not auto-generate chart: {exc}")
        logger.info("Run manually: python training/visualize_training.py --save")

    # Register version
    from utils.model_versioning import register_version

    final_val_accuracy = history_phase2.history.get("val_accuracy", [None])[-1]
    version_id = register_version(
        model_name=f"disease_model_{crop}",
        live_file_paths=[crop_cfg["path"], crop_cfg["labels_path"]],
        metrics={"final_val_accuracy": round(float(final_val_accuracy), 4) if final_val_accuracy else None},
        algorithm="EfficientNetB0",
    )
    logger.info(f"Registered as version {version_id}.")
    logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}" if final_val_accuracy else "No val accuracy recorded.")

    # Print final summary
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE - Disease Detection Model ({crop.upper()})")
    print(f"{'='*65}")
    print(f"  Crop:             {crop}")
    print(f"  Best model:         {model_path}")
    print(f"  All checkpoints:    {checkpoints_dir}")
    print(f"  Training history:   {history_path}")
    print(f"  Gradient reports:   {gradient_path}")
    print(f"  Training chart:     models/{crop}_disease_training_chart.png")
    print(f"  Labels:             {labels_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
