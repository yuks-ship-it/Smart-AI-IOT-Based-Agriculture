"""
Disease Model Training Diagnostic Script.

Run this BEFORE training to check:
1. Dataset folder structure
2. Image counts per class
3. Image readability
4. Class balance
5. Sample image visualization info

Usage:
    python training/diagnose_disease_data.py --data-dir "D:\RiceDisease"
    python training/diagnose_disease_data.py --data-dir "G:\My Drive\RiceDisease"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))


def check_dataset(data_dir: str):
    data_path = Path(data_dir)

    print("=" * 60)
    print("DISEASE DATASET DIAGNOSTIC REPORT")
    print("=" * 60)

    # ── 1. Check folder exists ────────────────────────────────────
    print(f"\n1. DATA DIRECTORY")
    print(f"   Path : {data_path}")
    print(f"   Exists: {data_path.exists()}")

    if not data_path.exists():
        print("\n   ERROR: Directory does not exist!")
        print("   Check your --data-dir path carefully.")
        print("   Run this in PowerShell to see your folders:")
        print(f'   dir "{data_path.parent}"')
        return

    # ── 2. List all subfolders ────────────────────────────────────
    print(f"\n2. SUBFOLDERS FOUND")
    subdirs = [d for d in sorted(data_path.iterdir()) if d.is_dir()]

    if not subdirs:
        print("   ERROR: No subfolders found in data directory!")
        print("   Your folder must contain class subfolders like:")
        print("     bacterial_leaf_blight/")
        print("     brown_spot/")
        print("     ...")
        return

    for d in subdirs:
        print(f"   {d.name}/")

    # ── 3. Count images per class ─────────────────────────────────
    print(f"\n3. IMAGE COUNTS PER CLASS")
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    class_counts = {}
    total_images = 0
    problem_classes = []

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue

        all_files = list(class_dir.iterdir())
        images = [
            f for f in all_files
            if f.suffix.lower() in VALID_EXTENSIONS
        ]
        non_images = [
            f for f in all_files
            if f.suffix.lower() not in VALID_EXTENSIONS and f.is_file()
        ]

        class_counts[class_dir.name] = len(images)
        total_images += len(images)

        status = "✓" if len(images) >= 50 else "⚠ LOW"
        print(f"   {class_dir.name:30s} {len(images):5d} images  {status}")

        if non_images:
            print(f"     (also found {len(non_images)} non-image files: "
                  f"{[f.suffix for f in non_images[:3]]})")

        if len(images) < 50:
            problem_classes.append(class_dir.name)

    print(f"\n   TOTAL: {total_images} images across {len(class_counts)} classes")

    # ── 4. Check class balance ────────────────────────────────────
    print(f"\n4. CLASS BALANCE CHECK")
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        ratio = max_count / max(min_count, 1)

        print(f"   Min images in a class: {min_count}")
        print(f"   Max images in a class: {max_count}")
        print(f"   Imbalance ratio: {ratio:.1f}x")

        if ratio > 5:
            print("   ⚠ WARNING: Dataset is very imbalanced!")
            print("   The model may ignore minority classes.")
        else:
            print("   ✓ Class balance is acceptable")

    # ── 5. Check image readability ────────────────────────────────
    print(f"\n5. IMAGE READABILITY CHECK (first 3 images per class)")
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        print("   Pillow not installed — skipping readability check")
        print("   Run: pip install Pillow")
        pil_available = False

    if pil_available:
        corrupt_count = 0
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in VALID_EXTENSIONS
            ][:3]

            class_ok = True
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"   CORRUPT: {img_path.name} — {e}")
                    corrupt_count += 1
                    class_ok = False

            if class_ok and images:
                # Check actual image size
                try:
                    with Image.open(images[0]) as img:
                        print(f"   {class_dir.name:30s} OK  "
                              f"size={img.size} mode={img.mode}")
                except Exception:
                    pass

        if corrupt_count == 0:
            print(f"\n   ✓ All sampled images are readable")
        else:
            print(f"\n   ⚠ Found {corrupt_count} corrupt images")

    # ── 6. Check folder naming ────────────────────────────────────
    print(f"\n6. FOLDER NAME CHECK")
    EXPECTED_CLASSES = [
        "bacterial_leaf_blight",
        "brown_spot",
        "leaf_blast",
        "panicle_blast",
        "narrow_brown_spot",
        "healthy",
    ]

    found_names = [d.name for d in data_path.iterdir() if d.is_dir()]

    for expected in EXPECTED_CLASSES:
        if expected in found_names:
            print(f"   ✓ {expected}")
        else:
            # Try to find a close match
            close = [n for n in found_names
                     if expected.replace("_", "").lower() in
                     n.replace(" ", "").replace("_", "").lower()
                     or n.replace(" ", "").replace("_", "").lower() in
                     expected.replace("_", "").lower()]
            if close:
                print(f"   ✗ {expected}  ← NOT FOUND")
                print(f"     Did you mean: {close[0]}  (needs renaming)")
            else:
                print(f"   ✗ {expected}  ← NOT FOUND, no close match")

    extra = [n for n in found_names if n not in EXPECTED_CLASSES]
    if extra:
        print(f"\n   Extra folders (will be included in training):")
        for e in extra:
            print(f"   + {e}")

    # ── 7. Summary and recommendations ───────────────────────────
    print(f"\n7. SUMMARY AND RECOMMENDATIONS")
    issues = []

    if total_images < 500:
        issues.append(
            f"Very few images ({total_images} total). "
            f"Need at least 100 per class for decent accuracy."
        )

    if problem_classes:
        issues.append(
            f"Low image count in: {problem_classes}. "
            f"Add more images to these classes."
        )

    wrong_names = [
        n for n in found_names if n not in EXPECTED_CLASSES
    ]
    if wrong_names:
        issues.append(
            f"Unexpected folder names: {wrong_names}. "
            f"Rename them to match expected class names."
        )

    if not issues:
        print("   ✓ Dataset looks good. Ready to train.")
        print(f"\n   Run training with:")
        print(f"   python training\\train_disease_model.py "
              f"--data-dir \"{data_dir}\"")
    else:
        print("   Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose disease training dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to disease image folder"
    )
    args = parser.parse_args()
    check_dataset(args.data_dir)


if __name__ == "__main__":
    main()