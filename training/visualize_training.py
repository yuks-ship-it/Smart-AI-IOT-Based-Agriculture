"""
Visualize Disease Model Training Results.

Generates comparison charts of Training vs Validation (Accuracy & Loss)
from the saved training history JSON file.

Usage:
    python training/visualize_training.py
    python training/visualize_training.py --history models/disease_training_history.json
    python training/visualize_training.py --save  (saves chart as PNG instead of showing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_history(history_path: str) -> dict:
    """Load training history from JSON file."""
    path = Path(history_path)
    if not path.exists():
        print(f"ERROR: Training history file not found at: {path}")
        print("Run the training script first, or provide the correct path with --history")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_charts(history: dict, save_path: str | None = None, model_name: str = "Disease Detection"):
    """Generate Training vs Validation comparison charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.size'] = 11
        matplotlib.rcParams['axes.titlesize'] = 13
        matplotlib.rcParams['axes.labelsize'] = 11
    except ImportError:
        print("ERROR: matplotlib is required. Install it with:")
        print("  pip install matplotlib")
        sys.exit(1)

    epochs = range(1, len(history["accuracy"]) + 1)

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Model - Training Log", fontsize=15, fontweight="bold")

    # ===== Accuracy Chart =====
    ax1.plot(epochs, history["accuracy"], "b-o", linewidth=2, markersize=4, label="Training Accuracy")
    ax1.plot(epochs, history["val_accuracy"], "r-s", linewidth=2, markersize=4, label="Validation Accuracy")

    # Highlight best validation accuracy
    best_val_acc = max(history["val_accuracy"])
    best_epoch = history["val_accuracy"].index(best_val_acc) + 1
    ax1.axhline(y=best_val_acc, color="green", linestyle="--", alpha=0.5, label=f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    ax1.scatter([best_epoch], [best_val_acc], color="green", s=100, zorder=5, marker="*")

    ax1.set_title("Accuracy: Training vs Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Add phase divider line
    if len(epochs) > 8:
        phase1_end = len(epochs) // 3
        if phase1_end < len(epochs):
            ax1.axvline(x=phase1_end + 0.5, color="purple", linestyle=":", alpha=0.6)
            ax1.text(phase1_end + 0.7, 0.15, "Phase 1 | Phase 2", color="purple", fontsize=9, alpha=0.7)

    # ===== Loss Chart =====
    ax2.plot(epochs, history["loss"], "b-o", linewidth=2, markersize=4, label="Training Loss")
    ax2.plot(epochs, history["val_loss"], "r-s", linewidth=2, markersize=4, label="Validation Loss")

    # Highlight best validation loss
    best_val_loss = min(history["val_loss"])
    best_loss_epoch = history["val_loss"].index(best_val_loss) + 1
    ax2.axhline(y=best_val_loss, color="green", linestyle="--", alpha=0.5, label=f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_loss_epoch})")
    ax2.scatter([best_loss_epoch], [best_val_loss], color="green", s=100, zorder=5, marker="*")

    ax2.set_title("Loss: Training vs Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Add phase divider line
    if len(epochs) > 8:
        phase1_end = len(epochs) // 3
        if phase1_end < len(epochs):
            ax2.axvline(x=phase1_end + 0.5, color="purple", linestyle=":", alpha=0.6)

    plt.tight_layout()

    # ===== Print Summary Table =====
    print("\n" + "=" * 65)
    print(f"  {model_name} Model - Training Summary")
    print("=" * 65)
    print(f"  Total Epochs Trained:     {len(epochs)}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}  (Epoch {best_epoch})")
    print(f"  Best Validation Loss:     {best_val_loss:.4f}  (Epoch {best_loss_epoch})")
    print(f"  Final Training Accuracy:  {history['accuracy'][-1]:.4f}")
    print(f"  Final Validation Accuracy:{history['val_accuracy'][-1]:.4f}")
    print(f"  Overfit Gap (Train-Val):  {history['accuracy'][-1] - history['val_accuracy'][-1]:.4f}")
    print("=" * 65)

    # Overfitting assessment
    gap = history["accuracy"][-1] - history["val_accuracy"][-1]
    if gap > 0.15:
        print("  WARNING: Significant overfitting detected (gap > 15%).")
        print("  Consider: more data, stronger augmentation, or more dropout.")
    elif gap > 0.08:
        print("  NOTE: Mild overfitting (gap 8-15%). Model is acceptable.")
    else:
        print("  GOOD: Minimal overfitting. Model generalizes well.")
    print("")

    # Epoch-by-epoch table
    print(f"  {'Epoch':<7} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
    print(f"  {'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i, ep in enumerate(epochs):
        print(
            f"  {ep:<7} {history['accuracy'][i]:<12.4f} {history['val_accuracy'][i]:<12.4f} "
            f"{history['loss'][i]:<12.4f} {history['val_loss'][i]:<12.4f}"
        )
    print("")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gradient_report(gradient_report_path: str, save_path: str | None = None):
    """Generate a gradient norms bar chart from gradient reports JSON."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    with open(gradient_report_path, "r", encoding="utf-8") as f:
        reports = json.load(f)

    if not reports:
        print("No gradient reports found.")
        return

    fig, axes = plt.subplots(1, len(reports), figsize=(7 * len(reports), 6))
    if len(reports) == 1:
        axes = [axes]

    fig.suptitle("Gradient Norms per Epoch", fontsize=14, fontweight="bold")

    for ax, report in zip(axes, reports):
        epoch = report["epoch"]
        top_layers = report.get("top_10_layers", {})

        if not top_layers:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Shorten layer names for readability
        names = [n.split("/")[-1][:25] for n in top_layers.keys()]
        values = list(top_layers.values())

        colors = []
        for v in values:
            if v < 1e-7:
                colors.append("red")      # vanishing
            elif v > 10.0:
                colors.append("orange")   # exploding
            else:
                colors.append("green")    # healthy

        ax.barh(names, values, color=colors)
        ax.set_xlabel("Gradient Norm")
        ax.set_title(f"Epoch {epoch}\nMean: {report['mean_norm']:.6f}")
        ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Gradient chart saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def list_checkpoints(checkpoints_dir: str):
    """List all saved epoch checkpoints with their validation accuracy."""
    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists():
        print(f"No checkpoints directory found at: {ckpt_path}")
        return

    checkpoints = sorted(ckpt_path.glob("epoch_*.*"))
    checkpoints = [c for c in checkpoints if c.suffix in (".keras", ".h5")]
    if not checkpoints:
        print("No checkpoint files found.")
        return

    print(f"\n{'='*60}")
    print(f"  Saved Checkpoints ({len(checkpoints)} total)")
    print(f"{'='*60}")
    print(f"  {'File':<45} {'Val Accuracy'}")
    print(f"  {'-'*45} {'-'*12}")

    for ckpt in checkpoints:
        name = ckpt.name
        # Extract val_acc from filename pattern: epoch_XX_val_acc_0.XXXX.h5
        try:
            val_acc = name.split("val_acc_")[1].replace(".h5", "")
            print(f"  {name:<45} {val_acc}")
        except (IndexError, ValueError):
            print(f"  {name:<45} {'?'}")

    print(f"{'='*60}")
    print(f"\n  To load a specific checkpoint for inference/comparison:")
    print(f"    import tensorflow as tf")
    print(f"    model = tf.keras.models.load_model('{checkpoints[0]}')")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Visualize disease model training results.")
    parser.add_argument(
        "--history", type=str, default="models/disease_training_history.json",
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save chart as PNG instead of showing interactively"
    )
    parser.add_argument(
        "--output", type=str, default="models/disease_training_chart.png",
        help="Output PNG path (used with --save)"
    )
    parser.add_argument(
        "--name", type=str, default="Disease Detection",
        help="Model name for chart title"
    )
    parser.add_argument(
        "--gradients", action="store_true",
        help="Also plot gradient report chart"
    )
    parser.add_argument(
        "--checkpoints", action="store_true",
        help="List all saved epoch checkpoints"
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    from utils.config_loader import resolve_path

    # List checkpoints if requested
    if args.checkpoints:
        list_checkpoints(str(resolve_path("models/checkpoints/disease")))

    # Plot training history
    history_path = resolve_path(args.history)
    output_path = resolve_path(args.output) if args.save else None

    history = load_history(str(history_path))
    plot_training_charts(history, save_path=str(output_path) if output_path else None, model_name=args.name)

    # Plot gradient reports if requested
    if args.gradients:
        grad_path = resolve_path("models/disease_gradient_reports.json")
        if grad_path.exists():
            grad_chart_path = str(resolve_path("models/disease_gradient_chart.png")) if args.save else None
            plot_gradient_report(str(grad_path), save_path=grad_chart_path)
        else:
            print("No gradient reports found. Retrain with gradient monitoring enabled.")


if __name__ == "__main__":
    main()
