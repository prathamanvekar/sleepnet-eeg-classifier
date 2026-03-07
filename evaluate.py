"""
evaluate.py
===========
Comprehensive evaluation with visualizations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)
import os

from dataset import SleepDataset
from model import SleepNet


# Configuration
DATA_DIR = "data/processed"
MODEL_PATH = "sleepnet.pth"
NUM_CLASSES = 5
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES
    )
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_hypnogram(y_true, y_pred, n_epochs=300, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    n_epochs = min(n_epochs, len(y_true))
    time_hours = np.arange(n_epochs) * 30 / 3600

    # True
    axes[0].step(time_hours, y_true[:n_epochs], 'b-', where='mid', linewidth=1)
    axes[0].set_ylabel('Sleep Stage')
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(STAGE_NAMES)
    axes[0].set_title('True Hypnogram', fontsize=12)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    # Predicted
    axes[1].step(time_hours, y_pred[:n_epochs], 'r-', where='mid', linewidth=1)
    axes[1].set_ylabel('Sleep Stage')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].set_yticklabels(STAGE_NAMES)
    axes[1].set_title('Predicted Hypnogram', fontsize=12)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_per_class_metrics(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy (recall)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Per-class precision
    per_class_prec = cm.diagonal() / (cm.sum(axis=0) + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    # Recall
    bars1 = axes[0].bar(STAGE_NAMES, per_class_acc, color=colors)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel('Recall', fontsize=12)
    axes[0].set_title('Per-Class Recall', fontsize=14)
    for bar, val in zip(bars1, per_class_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                     ha='center', fontsize=10)

    # Precision
    bars2 = axes[1].bar(STAGE_NAMES, per_class_prec, color=colors)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Per-Class Precision', fontsize=14)
    for bar, val in zip(bars2, per_class_prec):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                     ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def main():
    print("=" * 60)
    print("SLEEP STAGE CLASSIFICATION - EVALUATION")
    print("=" * 60)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Load data
    print("\nLoading data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print(f"Total samples: {len(y)}")

    # Dataset and loader
    dataset = SleepDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SleepNet(n_classes=NUM_CLASSES).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        print("Please train first: python train.py")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded: {MODEL_PATH}")

    # Evaluate
    print("\nEvaluating...")
    y_pred, y_true = evaluate_model(model, dataloader, device)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (Macro):  {f1_macro:.4f}")
    print(f"F1 Score (Weight): {f1_weighted:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=STAGE_NAMES))

    # Visualizations
    print("\nGenerating visualizations...")

    plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png")
    plot_per_class_metrics(y_true, y_pred, save_path="results/per_class_metrics.png")
    plot_hypnogram(y_true, y_pred, n_epochs=300, save_path="results/hypnogram.png")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Results saved to: results/")


if __name__ == "__main__":
    main()
