"""
Detailed Epoch-by-Epoch Training Report.

Shows for EACH epoch:
- Which architecture blocks are active (frozen vs trainable)
- Layer-level detail of trainable parameters
- Metrics (accuracy, loss, learning rate)
- Gradient flow information (if available)
- Clear visualization of phase transitions

Usage:
    python training/epoch_detail_report.py
    python training/epoch_detail_report.py --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ============================================================================
# EfficientNetB0 Architecture Reference
# ============================================================================
EFFICIENTNET_B0_BLOCKS = {
    "stem": {
        "layers": "stem_conv, stem_bn, stem_activation",
        "params": "~1,200",
        "output_shape": "112x112x32",
        "description": "Initial convolution: 3x3 Conv2D(32) + BatchNorm + Swish",
    },
    "block1": {
        "layers": "block1a_dwconv, block1a_bn, block1a_activation, block1a_se_*, block1a_project_*",
        "params": "~2,000",
        "output_shape": "112x112x16",
        "description": "MBConv1 (k3x3, expand=1): Depthwise Conv + SE + Project",
    },
    "block2": {
        "layers": "block2a_*, block2b_*",
        "params": "~10,000",
        "output_shape": "56x56x24",
        "description": "MBConv6 (k3x3, expand=6, stride=2): 2 sub-blocks with residual",
    },
    "block3": {
        "layers": "block3a_*, block3b_*",
        "params": "~15,000",
        "output_shape": "28x28x40",
        "description": "MBConv6 (k5x5, expand=6, stride=2): 2 sub-blocks with residual",
    },
    "block4": {
        "layers": "block4a_*, block4b_*, block4c_*",
        "params": "~40,000",
        "output_shape": "14x14x80",
        "description": "MBConv6 (k3x3, expand=6, stride=2): 3 sub-blocks with residual",
    },
    "block5": {
        "layers": "block5a_*, block5b_*, block5c_*",
        "params": "~75,000",
        "output_shape": "14x14x112",
        "description": "MBConv6 (k5x5, expand=6): 3 sub-blocks with residual",
    },
    "block6": {
        "layers": "block6a_*, block6b_*, block6c_*, block6d_*",
        "params": "~150,000",
        "output_shape": "7x7x192",
        "description": "MBConv6 (k5x5, expand=6, stride=2): 4 sub-blocks with residual",
    },
    "block7": {
        "layers": "block7a_*",
        "params": "~100,000",
        "output_shape": "7x7x320",
        "description": "MBConv6 (k3x3, expand=6): 1 sub-block, final features",
    },
    "top": {
        "layers": "top_conv, top_bn, top_activation",
        "params": "~410,000",
        "output_shape": "7x7x1280",
        "description": "Final 1x1 Conv(1280) + BatchNorm + Swish — feature expansion",
    },
}

CUSTOM_HEAD = {
    "global_avg_pool": {
        "params": "0",
        "output_shape": "1280",
        "description": "GlobalAveragePooling2D — collapses spatial dims",
    },
    "batch_norm_1": {
        "params": "~5,120",
        "output_shape": "1280",
        "description": "BatchNormalization — stabilizes activations",
    },
    "dropout_1": {
        "params": "0",
        "output_shape": "1280",
        "description": "Dropout(0.4) — regularization, drops 40% of neurons",
    },
    "dense_256": {
        "params": "~328,000 (1280*256 + 256 bias)",
        "output_shape": "256",
        "description": "Dense(256, relu) — learned feature combination",
    },
    "batch_norm_2": {
        "params": "~1,024",
        "output_shape": "256",
        "description": "BatchNormalization — stabilizes dense output",
    },
    "dropout_2": {
        "params": "0",
        "output_shape": "256",
        "description": "Dropout(0.2) — light regularization",
    },
    "dense_output": {
        "params": "~1,542 (256*6 + 6 bias)",
        "output_shape": "6",
        "description": "Dense(6, softmax) — final classification probabilities",
    },
}


def get_phase_config(epoch: int, phase1_epochs: int = 8):
    """Determine which layers are frozen/trainable at a given epoch."""
    if epoch <= phase1_epochs:
        return {
            "phase": "Phase 1 — Feature Extraction (Base Frozen)",
            "optimizer": "Adam",
            "loss": "CategoricalCrossentropy (label_smoothing=0.1)",
            "class_weights": "Yes (compensates panicle_blast underrepresentation)",
            "frozen_blocks": ["stem", "block1", "block2", "block3", "block4", "block5", "block6", "block7", "top"],
            "trainable_blocks": [],
            "trainable_head": ["global_avg_pool", "batch_norm_1", "dropout_1", "dense_256", "batch_norm_2", "dropout_2", "dense_output"],
            "total_trainable_params": "~335,686",
            "total_frozen_params": "~4,049,571",
            "strategy": "Only train the custom head. Pretrained EfficientNet features are used as-is.",
        }
    else:
        return {
            "phase": "Phase 2 — Fine-Tuning (Top 20 Layers Unfrozen)",
            "optimizer": "Adam",
            "loss": "CategoricalCrossentropy (label_smoothing=0.1)",
            "class_weights": "Yes",
            "frozen_blocks": ["stem", "block1", "block2", "block3", "block4", "block5"],
            "trainable_blocks": ["block6 (partial)", "block7", "top"],
            "trainable_head": ["global_avg_pool", "batch_norm_1", "dropout_1", "dense_256", "batch_norm_2", "dropout_2", "dense_output"],
            "total_trainable_params": "~995,686 (head + top 20 EfficientNet layers)",
            "total_frozen_params": "~3,389,571",
            "strategy": "Fine-tune top EfficientNet blocks to adapt pretrained features to rice disease patterns.",
        }


def print_epoch_detail(epoch: int, history: dict, gradient_reports: list, phase1_epochs: int = 8):
    """Print detailed block-level report for a single epoch."""
    idx = epoch - 1
    config = get_phase_config(epoch, phase1_epochs)

    acc = history["accuracy"][idx]
    val_acc = history["val_accuracy"][idx]
    loss = history["loss"][idx]
    val_loss = history["val_loss"][idx]
    lr = history["learning_rate"][idx]

    # Determine if this is a transition epoch
    is_phase_transition = (epoch == phase1_epochs + 1)
    is_lr_change = (idx > 0 and abs(lr - history["learning_rate"][idx - 1]) > 1e-8)

    print(f"\n{'#'*80}")
    print(f"##  EPOCH {epoch:>2}  {'|  *** PHASE TRANSITION ***' if is_phase_transition else ''}")
    print(f"{'#'*80}")

    # Phase info
    print(f"\n  PHASE: {config['phase']}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Optimizer: {config['optimizer']} | LR: {lr:.10f}{'  <-- LR REDUCED' if is_lr_change else ''}")
    print(f"  Loss Fn: {config['loss']}")
    print(f"  Class Weights: {config['class_weights']}")
    print(f"  Trainable Params: {config['total_trainable_params']}")
    print(f"  Frozen Params: {config['total_frozen_params']}")

    # Architecture visualization
    print(f"\n  {'='*70}")
    print(f"  ARCHITECTURE STATE AT EPOCH {epoch}")
    print(f"  {'='*70}")
    print(f"  Input: 224 x 224 x 3 (RGB image, preprocessed by EfficientNet preprocess_input)")
    print(f"  {'─'*70}")

    # EfficientNet blocks
    print(f"\n  ┌─── EfficientNetB0 Backbone ──────────────────────────────────────────┐")
    for block_name, block_info in EFFICIENTNET_B0_BLOCKS.items():
        is_frozen = block_name in config["frozen_blocks"]
        is_trainable = block_name in [b.split(" ")[0] for b in config["trainable_blocks"]]
        
        status = "FROZEN" if is_frozen else "TRAINABLE"
        icon = "❄" if is_frozen else "🔥" if not is_frozen else "  "
        color_indicator = "[ FROZEN ]" if is_frozen else "[TRAINING]"

        print(f"  │  {color_indicator} {block_name:<8} → {block_info['output_shape']:<14} "
              f"| {block_info['params']:<12} | {block_info['description'][:40]}")
    
    print(f"  └─────────────────────────────────────────────────────────────────────┘")

    # Custom head
    print(f"\n  ┌─── Custom Classification Head ─────────────────────────────────────┐")
    for layer_name, layer_info in CUSTOM_HEAD.items():
        print(f"  │  [TRAINING] {layer_name:<14} → {layer_info['output_shape']:<10} "
              f"| {layer_info['params']:<28} | {layer_info['description'][:35]}")
    print(f"  └─────────────────────────────────────────────────────────────────────┘")

    print(f"\n  Output: 6 classes [bacterial_leaf_blight, brown_spot, healthy,")
    print(f"                      leaf_blast, narrow_brown_spot, panicle_blast]")

    # Metrics
    print(f"\n  {'─'*70}")
    print(f"  METRICS")
    print(f"  {'─'*70}")
    print(f"  Training Accuracy:    {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Validation Accuracy:  {val_acc:.4f}  ({val_acc*100:.2f}%)")
    print(f"  Training Loss:        {loss:.4f}")
    print(f"  Validation Loss:      {val_loss:.4f}")
    print(f"  Learning Rate:        {lr:.10f}")
    gap = acc - val_acc
    print(f"  Overfit Gap:          {gap:.4f}  ({gap*100:.2f}%)"
          f"{'  ⚠ OVERFITTING' if gap > 0.1 else '  ✓ OK' if gap < 0.08 else '  ~ MILD'}")

    # Change from previous epoch
    if idx > 0:
        prev_acc = history["accuracy"][idx - 1]
        prev_val_acc = history["val_accuracy"][idx - 1]
        prev_loss = history["loss"][idx - 1]
        prev_val_loss = history["val_loss"][idx - 1]

        print(f"\n  CHANGE FROM EPOCH {epoch-1}:")
        print(f"  Train Acc:  {acc - prev_acc:+.4f}  {'↑' if acc > prev_acc else '↓'}")
        print(f"  Val Acc:    {val_acc - prev_val_acc:+.4f}  {'↑' if val_acc > prev_val_acc else '↓'}")
        print(f"  Train Loss: {loss - prev_loss:+.4f}  {'↓ (better)' if loss < prev_loss else '↑ (worse)'}")
        print(f"  Val Loss:   {val_loss - prev_val_loss:+.4f}  {'↓ (better)' if val_loss < prev_val_loss else '↑ (worse)'}")

    # Phase transition detail
    if is_phase_transition:
        print(f"\n  {'*'*70}")
        print(f"  *** PHASE TRANSITION: Phase 1 → Phase 2 ***")
        print(f"  {'*'*70}")
        print(f"  WHAT CHANGED:")
        print(f"    - Top 20 layers of EfficientNet UNFROZEN (were frozen)")
        print(f"    - Blocks now trainable: block6 (partial), block7, top")
        print(f"    - Learning rate RESET to 5e-5 (was {history['learning_rate'][idx-1]:.8f})")
        print(f"    - Trainable params: ~335K → ~995K (3x increase)")
        print(f"    - Model now adapts pretrained features to disease patterns")
        print(f"  {'*'*70}")

    # Gradient info if available
    grad_report = None
    for r in gradient_reports:
        if r["epoch"] == epoch:
            grad_report = r
            break

    if grad_report:
        print(f"\n  {'─'*70}")
        print(f"  GRADIENT FLOW (sampled from validation batches)")
        print(f"  {'─'*70}")
        print(f"  Mean Gradient Norm:  {grad_report['mean_norm']:.8f}")
        print(f"  Max Gradient Norm:   {grad_report['max_norm']:.8f}")
        print(f"  Min Gradient Norm:   {grad_report['min_norm']:.8f}")
        print(f"  Healthy Layers:      {grad_report['healthy_layers']}")
        print(f"  Vanishing Layers:    {grad_report['vanishing_layers']}")
        print(f"  Exploding Layers:    {grad_report['exploding_layers']}")
        print(f"\n  Layer-wise Gradients:")
        for layer_name, norm in grad_report.get("top_10_layers", {}).items():
            bar_len = min(int(norm * 10), 50)
            bar = "█" * bar_len
            status = "VANISHING" if norm < 1e-7 else "EXPLODING" if norm > 10 else "OK"
            print(f"    {layer_name:<20} {norm:>10.6f}  {bar}  [{status}]")


def print_summary_table(history: dict, phase1_epochs: int = 8):
    """Print a comparison summary showing all epochs side by side."""
    total_epochs = len(history["accuracy"])

    print(f"\n\n{'='*100}")
    print(f"  EPOCH-BY-EPOCH SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"  {'Ep':<4} {'Phase':<8} {'Train Acc':<11} {'Val Acc':<11} "
          f"{'Train Loss':<12} {'Val Loss':<12} {'LR':<14} {'Gap':<8} {'Note'}")
    print(f"  {'─'*4} {'─'*8} {'─'*11} {'─'*11} {'─'*12} {'─'*12} {'─'*14} {'─'*8} {'─'*20}")

    best_val_acc = 0
    best_epoch = 0

    for i in range(total_epochs):
        epoch = i + 1
        phase = "P1" if epoch <= phase1_epochs else "P2"
        acc = history["accuracy"][i]
        val_acc = history["val_accuracy"][i]
        loss = history["loss"][i]
        val_loss = history["val_loss"][i]
        lr = history["learning_rate"][i]
        gap = acc - val_acc

        note = ""
        if epoch == phase1_epochs + 1:
            note = "← PHASE SWITCH"
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            note = "← BEST VAL ACC"
        if i > 0 and abs(lr - history["learning_rate"][i-1]) > 1e-8:
            note = note or "← LR REDUCED"

        print(f"  {epoch:<4} {phase:<8} {acc:<11.4f} {val_acc:<11.4f} "
              f"{loss:<12.4f} {val_loss:<12.4f} {lr:<14.10f} {gap:<8.4f} {note}")

    print(f"\n  Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")
    print(f"{'='*100}")


def generate_visual_timeline(history: dict, phase1_epochs: int = 8):
    """Generate an ASCII art timeline showing training progression."""
    total_epochs = len(history["accuracy"])

    print(f"\n\n{'='*80}")
    print(f"  TRAINING TIMELINE (ASCII Visualization)")
    print(f"{'='*80}")

    # Accuracy timeline
    print(f"\n  Val Accuracy Progress:")
    print(f"  100%|", end="")
    print(f"{'─'*total_epochs*3}|")

    for target in [90, 80, 70, 60, 50]:
        line = f"  {target:>3}%|"
        for i in range(total_epochs):
            val_acc = history["val_accuracy"][i] * 100
            if val_acc >= target:
                if i + 1 == phase1_epochs:
                    line += " ‖ "  # phase boundary
                else:
                    line += " █ "
            else:
                if i + 1 == phase1_epochs:
                    line += " ‖ "
                else:
                    line += " · "
        line += "|"
        print(line)

    print(f"    0%|", end="")
    print(f"{'─'*total_epochs*3}|")
    print(f"       ", end="")
    for i in range(total_epochs):
        print(f"{i+1:>3}", end="")
    print(f"  ← Epoch")
    print(f"       {'Phase 1 (Frozen)':^{phase1_epochs*3}}{'Phase 2 (Fine-tune)':^{(total_epochs-phase1_epochs)*3}}")

    # Learning rate timeline
    print(f"\n  Learning Rate Decay:")
    lrs = history["learning_rate"]
    max_lr = max(lrs)
    for i, lr in enumerate(lrs):
        bar_len = max(1, int((lr / max_lr) * 30))
        phase = "P1" if i + 1 <= phase1_epochs else "P2"
        print(f"  Ep{i+1:>2} [{phase}] {'▓' * bar_len}{'░' * (30 - bar_len)} LR={lr:.8f}")


def main():
    parser = argparse.ArgumentParser(description="Detailed epoch-by-epoch training report.")
    parser.add_argument("--history", type=str, default="models/disease_training_history.json")
    parser.add_argument("--gradients", type=str, default="models/disease_gradient_reports.json")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument("--output", type=str, default="models/disease_epoch_detail_report.txt")
    parser.add_argument("--phase1-epochs", type=int, default=8, help="Number of Phase 1 epochs")
    args = parser.parse_args()

    from utils.config_loader import resolve_path

    # Load data
    history_path = resolve_path(args.history)
    gradient_path = resolve_path(args.gradients)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    gradient_reports = []
    if gradient_path.exists():
        with open(gradient_path, "r", encoding="utf-8") as f:
            gradient_reports = json.load(f)

    total_epochs = len(history["accuracy"])

    # Redirect to file if saving
    original_stdout = sys.stdout
    if args.save:
        output_path = resolve_path(args.output)
        sys.stdout = open(str(output_path), "w", encoding="utf-8")

    # Print header
    print(f"{'#'*80}")
    print(f"##")
    print(f"##  DISEASE DETECTION MODEL — DETAILED EPOCH REPORT")
    print(f"##")
    print(f"##  Architecture: EfficientNetB0 + Custom Head")
    print(f"##  Total Parameters: ~4,385,257 (Backbone: 4,049,571 + Head: 335,686)")
    print(f"##  Input: 224x224x3 RGB images")
    print(f"##  Output: 6 classes (rice diseases)")
    print(f"##  Total Epochs: {total_epochs} (Phase 1: {args.phase1_epochs}, Phase 2: {total_epochs - args.phase1_epochs})")
    print(f"##")
    print(f"##  Phase 1: Base frozen, train head only (LR=1e-3)")
    print(f"##  Phase 2: Top 20 layers unfrozen (LR=5e-5, label smoothing, class weights)")
    print(f"##")
    print(f"{'#'*80}")

    # Print each epoch detail
    for epoch in range(1, total_epochs + 1):
        print_epoch_detail(epoch, history, gradient_reports, args.phase1_epochs)

    # Print summary table
    print_summary_table(history, args.phase1_epochs)

    # Print timeline
    generate_visual_timeline(history, args.phase1_epochs)

    # Restore stdout
    if args.save:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Report saved to: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
