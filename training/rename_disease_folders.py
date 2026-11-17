"""
Rename disease dataset folders to match project class names.

Renames your folders from human-readable names (Bacterial Blight)
to project class names (bacterial_leaf_blight) that match the
disease_treatment.json config keys.

Folders not in the rename map are left untouched and will be
automatically ignored during training.

Usage:
    python training/rename_disease_folders.py --data-dir "D:\\RiceDisease\\RiceDisease"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Rename map ─────────────────────────────────────────────────────────────
# Left  = your current folder name (exact, case-sensitive)
# Right = project class name (must match disease_treatment.json keys)
RENAME_MAP = {
    "Bacterial Blight":  "bacterial_leaf_blight",
    "Brown Spot":        "brown_spot",
    "Healthy":           "healthy",
    "Leaf Blast":        "leaf_blast",
    "Narrow Brown Spot": "narrow_brown_spot",
    "Neck Blast":        "panicle_blast",
}

# ── Folders to skip (not needed in this project) ──────────────────────────
SKIP_FOLDERS = [
    "Leaf Scald",
    "Leaf Smut",
    "Hispa",
    "Tungro",
    "Ragged Stunt Virus",
    "Grassy Stunt Virus",
    "Bakane",
]


def rename_folders(data_dir: str) -> None:
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return

    print("=" * 60)
    print("DISEASE FOLDER RENAME TOOL")
    print("=" * 60)
    print(f"\nData directory: {data_path}\n")

    subdirs = [d for d in sorted(data_path.iterdir()) if d.is_dir()]

    if not subdirs:
        print("ERROR: No subfolders found.")
        return

    renamed = []
    skipped = []
    already_correct = []
    unknown = []

    for folder in subdirs:
        name = folder.name

        # Already correctly named
        if name in RENAME_MAP.values():
            already_correct.append(name)
            print(f"  ✓ Already correct: {name}")
            continue

        # In rename map — rename it
        if name in RENAME_MAP:
            new_name = RENAME_MAP[name]
            new_path = data_path / new_name

            if new_path.exists():
                print(f"  ⚠ Target already exists: {new_name} — skipping rename of {name}")
                continue

            os.rename(folder, new_path)
            img_count = len(list(new_path.iterdir()))
            print(f"  ✓ Renamed: {name:25s} → {new_name}  ({img_count} images)")
            renamed.append(f"{name} → {new_name}")
            continue

        # In skip list — leave it alone
        if name in SKIP_FOLDERS:
            print(f"  ⊘ Skipped (not needed): {name}")
            skipped.append(name)
            continue

        # Unknown folder
        print(f"  ? Unknown folder: {name} — leaving untouched")
        unknown.append(name)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Renamed:         {len(renamed)}")
    for r in renamed:
        print(f"    {r}")

    print(f"\n  Already correct: {len(already_correct)}")
    for a in already_correct:
        print(f"    {a}")

    print(f"\n  Skipped:         {len(skipped)}")
    for s in skipped:
        print(f"    {s}")

    if unknown:
        print(f"\n  Unknown (not touched): {len(unknown)}")
        for u in unknown:
            print(f"    {u}")

    # ── Final folder list ─────────────────────────────────────────────────
    print(f"\nFinal folders in {data_path}:")
    for folder in sorted(data_path.iterdir()):
        if folder.is_dir():
            img_count = len([
                f for f in folder.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ])
            in_project = "✓" if folder.name in RENAME_MAP.values() else "⊘ not used"
            print(f"  {folder.name:30s} {img_count:5d} images  {in_project}")

    print("\n" + "=" * 60)
    print("NEXT STEP — Run training:")
    print(f'  python training\\train_disease_model.py --data-dir "{data_dir}"')
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rename disease folders to match project class names."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to disease image folder e.g. D:\\RiceDisease\\RiceDisease"
    )
    args = parser.parse_args()
    rename_folders(args.data_dir)


if __name__ == "__main__":
    main()