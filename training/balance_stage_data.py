"""
Balance Stage Dataset via Offline Augmentation.

Problem: Class imbalance in growth stage data:
  vegetative: 177, tillering: 157, maturity: 90, flowering: 44

Solution: Generate augmented copies of underrepresented classes
so all classes have approximately the same number of images.
Original images are preserved; augmented copies are added alongside them.

Run BEFORE training:
    python training/balance_stage_data.py

Then train as normal:
    python training/train_stage_model.py --epochs 30 --batch-size 16
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

STAGE_DATA_DIR = Path("datasets/crop_images/rice/stage")


def augment_image(img_array: np.ndarray, seed: int) -> np.ndarray:
    """Apply random augmentations to a single image array."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = img_array.shape[:2]

    # Random rotation (-30 to +30 degrees)
    angle = rng.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness (0.7 to 1.3)
    factor = rng.uniform(0.7, 1.3)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # Random horizontal flip (50% chance)
    if rng.random() > 0.5:
        img = cv2.flip(img, 1)

    # Random zoom (crop and resize back)
    zoom = rng.uniform(0.8, 1.2)
    if zoom < 1.0:
        # Zoom out — add border
        new_h, new_w = int(h * zoom), int(w * zoom)
        resized = cv2.resize(img, (new_w, new_h))
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        img = cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h,
                                 pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
    else:
        # Zoom in — crop center
        crop_h, crop_w = int(h / zoom), int(w / zoom)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
        img = cv2.resize(cropped, (w, h))

    # Random color shift
    shift = rng.integers(-20, 20, size=3)
    img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    return img


def balance_classes(data_dir: Path, target_count: int | None = None):
    """
    Balance all stage classes by augmenting smaller classes up to target_count.
    If target_count is None, uses the size of the largest class.
    """
    import cv2

    if not data_dir.exists():
        logger.error(f"Stage data directory not found: {data_dir}")
        return

    # Count images per class
    class_counts = {}
    for stage_dir in sorted(data_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        images = [f for f in stage_dir.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]
        class_counts[stage_dir.name] = len(images)

    if not class_counts:
        logger.error("No stage classes found!")
        return

    max_count = max(class_counts.values())
    target = target_count or max_count

    print(f"\n{'='*60}")
    print(f"  STAGE DATA BALANCING")
    print(f"{'='*60}")
    print(f"  Target count per class: {target}")
    print(f"\n  Current distribution:")
    for name, count in sorted(class_counts.items()):
        deficit = max(0, target - count)
        bar = "#" * min(count // 3, 50)
        status = f"(need +{deficit} augmented)" if deficit > 0 else "(OK)"
        print(f"    {name:<12} {count:>4} images {bar} {status}")

    # Generate augmented images for underrepresented classes
    print(f"\n  Augmenting...")
    total_generated = 0

    for stage_dir in sorted(data_dir.iterdir()):
        if not stage_dir.is_dir():
            continue

        original_images = [f for f in stage_dir.iterdir()
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                          and not f.stem.startswith("aug_")]
        current_count = len(list(stage_dir.iterdir()))
        # Count only non-augmented originals for source
        originals = [f for f in stage_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                     and not f.stem.startswith("aug_")]

        need = target - len(originals)
        if need <= 0:
            continue

        logger.info(f"  {stage_dir.name}: generating {need} augmented images from {len(originals)} originals")

        generated = 0
        seed = 42
        while generated < need:
            for orig_path in originals:
                if generated >= need:
                    break
                # Read original image
                img = cv2.imread(str(orig_path))
                if img is None:
                    continue

                # Generate augmented version
                aug_img = augment_image(img, seed)
                seed += 1

                # Save with aug_ prefix
                aug_name = f"aug_{generated:04d}_{orig_path.name}"
                aug_path = stage_dir / aug_name
                cv2.imwrite(str(aug_path), aug_img)
                generated += 1

        total_generated += generated
        print(f"    {stage_dir.name}: +{generated} augmented images created")

    # Final count
    print(f"\n  Final distribution:")
    for stage_dir in sorted(data_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        all_images = [f for f in stage_dir.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]
        originals = [f for f in all_images if not f.stem.startswith("aug_")]
        augmented = [f for f in all_images if f.stem.startswith("aug_")]
        total = len(all_images)
        bar = "#" * min(total // 3, 50)
        print(f"    {stage_dir.name:<12} {total:>4} images ({len(originals)} original + {len(augmented)} augmented) {bar}")

    print(f"\n  Total augmented images generated: {total_generated}")
    print(f"  All classes now balanced at ~{target} images each.")
    print(f"{'='*60}\n")


def clean_augmented(data_dir: Path):
    """Remove all previously generated augmented images (files starting with aug_)."""
    removed = 0
    for stage_dir in data_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        for f in stage_dir.iterdir():
            if f.stem.startswith("aug_"):
                f.unlink()
                removed += 1
    print(f"  Removed {removed} previously augmented images.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Balance stage dataset via offline augmentation.")
    parser.add_argument("--data-dir", type=str, default=str(STAGE_DATA_DIR),
                        help="Path to stage data directory")
    parser.add_argument("--target", type=int, default=None,
                        help="Target images per class (default: match largest class)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove previously augmented images before regenerating")
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    if args.clean:
        print("  Cleaning old augmented images...")
        clean_augmented(data_path)

    balance_classes(data_path, target_count=args.target)
