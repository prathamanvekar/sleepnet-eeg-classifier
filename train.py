"""
train.py
========
Training with class balancing and validation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import os

from dataset import SleepDataset
from model import SleepNet


# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 5
PATIENCE = 10

DATA_DIR = "data/processed"
SAVE_PATH = "sleepnet.pth"

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def create_weighted_sampler(labels):
    """Create sampler that balances class frequencies."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), acc, f1


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("SLEEP STAGE CLASSIFICATION - TRAINING")
    print("=" * 60)

    # Load data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    print(f"\nData: {X.shape}, Labels: {y.shape}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {STAGE_NAMES[cls]}: {cnt} ({100*cnt/len(y):.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Datasets
    train_ds = SleepDataset(X_train, y_train, augment=True)
    val_ds = SleepDataset(X_val, y_val)
    test_ds = SleepDataset(X_test, y_test)

    # Weighted sampler for balanced training
    train_sampler = create_weighted_sampler(y_train)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SleepNet(n_classes=NUM_CLASSES, dropout=0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for loss
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = torch.tensor(len(y_train) / (NUM_CLASSES * class_counts + 1e-6), dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training loop
    print("\n" + "-" * 60)
    print("Training...")
    print("-" * 60)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_dl, criterion, device)

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.1f}% F1={train_f1:.3f} | "
              f"Val: Acc={val_acc:.1f}% F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Saved best model (F1={val_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_dl, criterion, device)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test F1 (Macro): {test_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=STAGE_NAMES))

    print(f"\nModel saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
