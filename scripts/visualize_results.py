# visualize_results.py
# Visualize YOLOv8 training results: loss curves, mAP curves, confusion matrix
# Place this file in: detection_helmet/scripts/

import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
TRAIN_DIR     = BASE_DIR / "runs" / "detect" / "runs" / "train" / "helmet_v1"
RESULTS_CSV   = TRAIN_DIR / "results.csv"
CONFUSION_PNG = TRAIN_DIR / "confusion_matrix.png"
OUTPUT_DIR    = BASE_DIR / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Loss & mAP Curves
def plot_training_curves():
    if not RESULTS_CSV.exists():
        print(f"[WARNING] results.csv not found at: {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()
    epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("YOLOv8 Training Results – Helmet Detection",
                 fontsize=14, fontweight="bold")

    # Loss plots
    loss_cfg = [
        ("train/box_loss", "val/box_loss",  "Box Loss",            axes[0, 0]),
        ("train/cls_loss", "val/cls_loss",  "Classification Loss", axes[0, 1]),
        ("train/dfl_loss", "val/dfl_loss",  "DFL Loss",            axes[0, 2]),
    ]
    for train_col, val_col, title, ax in loss_cfg:
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], label="Train", color="blue",   linewidth=1.5)
        if val_col in df.columns:
            ax.plot(epochs, df[val_col],   label="Val",   color="orange", linewidth=1.5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

    # mAP / Precision / Recall plots
    metric_cfg = [
        ("metrics/mAP50(B)",    "mAP@0.5",       axes[1, 0], "green"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95",  axes[1, 1], "purple"),
        ("metrics/precision(B)","Precision",      axes[1, 2], "red"),
    ]
    for col, title, ax, color in metric_cfg:
        if col in df.columns:
            ax.plot(epochs, df[col], label=title, color=color, linewidth=1.5)
        if title == "Precision" and "metrics/recall(B)" in df.columns:
            ax.plot(epochs, df["metrics/recall(B)"],
                    label="Recall", color="brown", linewidth=1.5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Training curves  → {out}")


# 2. Confusion Matrix
def plot_confusion_matrix():
    if CONFUSION_PNG.exists():
        out = OUTPUT_DIR / "confusion_matrix.png"
        shutil.copy(CONFUSION_PNG, out)
        print(f"[SAVED] Confusion matrix → {out}")
    else:
        print(f"[WARNING] confusion_matrix.png not found at: {CONFUSION_PNG}")
        _placeholder_cm()


def _placeholder_cm():
    classes = ["helmet", "head", "person"]
    cm = np.array([[145, 8, 3], [6, 132, 5], [4, 7, 178]])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_title("Confusion Matrix (Placeholder)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    out = OUTPUT_DIR / "confusion_matrix_placeholder.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Placeholder CM   → {out}")


# 3. mAP Summary Bar Chart
def plot_map_summary():
    if not RESULTS_CSV.exists():
        return

    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()

    map_col = "metrics/mAP50(B)"
    if map_col not in df.columns:
        print("[WARNING] mAP column not found, skipping summary chart.")
        return

    best = df.loc[df[map_col].idxmax()]
    metrics = {
        "Precision":    best.get("metrics/precision(B)", 0),
        "Recall":       best.get("metrics/recall(B)", 0),
        "mAP@0.5":      best.get("metrics/mAP50(B)", 0),
        "mAP@0.5:0.95": best.get("metrics/mAP50-95(B)", 0),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics.keys(), metrics.values(),
                  color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Best Epoch – Model Performance Summary", fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = OUTPUT_DIR / "map_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] mAP summary      → {out}")


# Entry point
if __name__ == "__main__":
    print("=" * 55)
    print("  Helmet Detection – Training Visualization (YOLOv8)")
    print("=" * 55)
    plot_training_curves()
    plot_confusion_matrix()
    plot_map_summary()
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")