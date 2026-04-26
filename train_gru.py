#!/usr/bin/env python3
"""
BE124 Fall Prediction - GRU Baseline Model
============================================
Trains a GRU model on windowed IMU data for perturbation prediction.

Usage:
  # Train with default settings (100Hz, 500ms horizon)
  python train_gru.py --data-dir dataset_100hz/

  # Train all horizons
  python train_gru.py --data-dir dataset_100hz/ --horizons 100 200 300 500

  # Custom hyperparameters
  python train_gru.py --data-dir dataset_100hz/ --hidden 128 --layers 2 --epochs 100 --lr 0.001

  # Quick test
  python train_gru.py --data-dir dataset_100hz/ --epochs 5

Requirements:
  pip install torch numpy scikit-learn matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import time


# ============================================================
# Model
# ============================================================
class GRUClassifier(nn.Module):
    """
    GRU for binary time-series classification.
    Input: (batch, timesteps, features)
    Output: (batch, 1) probability
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.gru(x)           # out: (batch, timesteps, hidden)
        out = out[:, -1, :]            # take last timestep: (batch, hidden)
        out = self.dropout(out)
        out = self.fc(out)             # (batch, 1)
        return out.squeeze(-1)         # (batch,)


# ============================================================
# Data loading
# ============================================================
def load_data(data_dir, horizon_ms=500):
    """Load preprocessed numpy arrays."""
    data_dir = Path(data_dir)

    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')

    Y_train = np.load(data_dir / f'Y_train_{horizon_ms}ms.npy')
    Y_val = np.load(data_dir / f'Y_val_{horizon_ms}ms.npy')
    Y_test = np.load(data_dir / f'Y_test_{horizon_ms}ms.npy')

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def make_dataloader(X, Y, batch_size=64, oversample=True, shuffle=True):
    """Create DataLoader with optional oversampling for class imbalance."""
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)

    if oversample and shuffle:
        # Weighted sampler to handle class imbalance
        n_pos = Y.sum()
        n_neg = len(Y) - n_pos
        if n_pos > 0 and n_neg > 0:
            weight_pos = n_neg / n_pos
            weights = np.where(Y == 1, weight_pos, 1.0)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    n_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())
            n_batches += 1

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / n_batches,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'accuracy': (all_preds == all_labels).mean(),
    }

    # AUC if both classes present
    if len(np.unique(all_labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probs)
        prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
        metrics['auc_pr'] = auc(rec_curve, prec_curve)
    else:
        metrics['auc_roc'] = 0
        metrics['auc_pr'] = 0

    return metrics, all_preds, all_probs, all_labels


# ============================================================
# Main training loop
# ============================================================
def train_model(data_dir, horizon_ms=500, hidden_size=64, num_layers=2,
                dropout=0.3, lr=0.001, epochs=50, batch_size=64,
                patience=10, output_dir='results/'):
    """Full training pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Load data
    print(f"  Loading data from {data_dir}...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(data_dir, horizon_ms)

    n_features = X_train.shape[2]
    n_timesteps = X_train.shape[1]

    print(f"  Train: {X_train.shape} ({Y_train.sum():.0f} pos / {len(Y_train)} total)")
    print(f"  Val:   {X_val.shape} ({Y_val.sum():.0f} pos / {len(Y_val)} total)")
    print(f"  Test:  {X_test.shape} ({Y_test.sum():.0f} pos / {len(Y_test)} total)")

    # Normalize features (fit on train only)
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0)
    std[std < 1e-8] = 1.0  # prevent division by zero

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Save normalization params
    np.save(output_dir / 'norm_mean.npy', mean)
    np.save(output_dir / 'norm_std.npy', std)

    # DataLoaders
    train_loader = make_dataloader(X_train, Y_train, batch_size, oversample=True)
    val_loader = make_dataloader(X_val, Y_val, batch_size, oversample=False, shuffle=False)
    test_loader = make_dataloader(X_test, Y_test, batch_size, oversample=False, shuffle=False)

    # Model
    model = GRUClassifier(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print(f"\n  Model: GRU(features={n_features}, hidden={hidden_size}, "
          f"layers={num_layers}, dropout={dropout})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Loss with class weight
    n_pos = Y_train.sum()
    n_neg = len(Y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device) if n_pos > 0 else torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.1f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Training loop
    print(f"\n  Training for {epochs} epochs (patience={patience})...")
    print(f"  {'Epoch':>5s} {'Train Loss':>10s} {'Val Loss':>10s} {'Val F1':>8s} "
          f"{'Val Prec':>8s} {'Val Rec':>8s} {'Val AUC':>8s}")
    print(f"  {'-'*60}")

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [],
               'val_precision': [], 'val_recall': [], 'val_auc': []}
    best_f1 = 0
    best_epoch = 0
    no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_auc'].append(val_metrics['auc_pr'])

        scheduler.step(val_metrics['f1'])

        # Print every 5 epochs or if improved
        if epoch % 5 == 0 or val_metrics['f1'] > best_f1:
            marker = ' *' if val_metrics['f1'] > best_f1 else ''
            print(f"  {epoch:>5d} {train_loss:>10.4f} {val_metrics['loss']:>10.4f} "
                  f"{val_metrics['f1']:>8.4f} {val_metrics['precision']:>8.4f} "
                  f"{val_metrics['recall']:>8.4f} {val_metrics['auc_pr']:>8.4f}{marker}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    train_time = time.time() - start_time
    print(f"\n  Training done in {train_time:.1f}s")
    print(f"  Best val F1: {best_f1:.4f} at epoch {best_epoch}")

    # ============================================================
    # Test evaluation
    # ============================================================
    print(f"\n  Loading best model (epoch {best_epoch})...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))

    test_metrics, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — Horizon: {horizon_ms}ms")
    print(f"{'='*60}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {test_metrics['auc_pr']:.4f}")
    print(f"{'='*60}")

    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=['Normal', 'Perturbation'],
                                zero_division=0))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"    TN={cm[0,0]:>5d}  FP={cm[0,1]:>5d}")
    print(f"    FN={cm[1,0]:>5d}  TP={cm[1,1]:>5d}")

    # ============================================================
    # Plots
    # ============================================================

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Training History — GRU — {horizon_ms}ms Horizon', fontsize=13, fontweight='bold')

    axes[0].plot(history['train_loss'], label='Train', color='#534AB7')
    axes[0].plot(history['val_loss'], label='Val', color='#D85A30')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(history['val_f1'], label='F1', color='#534AB7')
    axes[1].plot(history['val_precision'], label='Precision', color='#1D9E75')
    axes[1].plot(history['val_recall'], label='Recall', color='#D85A30')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(history['val_auc'], label='AUC-PR', color='#534AB7')
    axes[2].set_title('Validation AUC-PR')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrix heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Perturbation'])
    ax.set_yticklabels(['Normal', 'Perturbation'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — GRU — {horizon_ms}ms')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Precision-Recall curve
    if len(np.unique(test_labels)) > 1:
        prec_curve, rec_curve, thresholds = precision_recall_curve(test_labels, test_probs)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(rec_curve, prec_curve, color='#534AB7', lw=2)
        ax.fill_between(rec_curve, prec_curve, alpha=0.2, color='#534AB7')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — AUC={test_metrics["auc_pr"]:.3f}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ============================================================
    # Save results
    # ============================================================
    results = {
        'model': 'GRU',
        'horizon_ms': horizon_ms,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'train_time_s': round(train_time, 1),
        'n_params': n_params,
        'input_shape': list(X_train.shape),
        'test_metrics': {k: round(float(v), 4) for k, v in test_metrics.items()},
        'confusion_matrix': cm.tolist(),
        'history': {k: [round(float(v), 4) for v in vals] for k, vals in history.items()},
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {output_dir}/:")
    print(f"    best_model.pt, results.json, training_curves.png,")
    print(f"    confusion_matrix.png, pr_curve.png, norm_mean/std.npy")

    return results


# ============================================================
# Multi-horizon experiment
# ============================================================
def run_all_horizons(data_dir, horizons, **kwargs):
    """Train models for multiple prediction horizons and compare."""
    all_results = {}

    for h in horizons:
        print(f"\n\n{'#'*60}")
        print(f"  HORIZON: {h}ms")
        print(f"{'#'*60}")

        out = Path(kwargs.get('output_dir', 'results')) / f'gru_{h}ms'
        results = train_model(data_dir, horizon_ms=h, output_dir=str(out), **{
            k: v for k, v in kwargs.items() if k != 'output_dir'
        })
        all_results[h] = results

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('GRU Performance vs Prediction Horizon', fontsize=14, fontweight='bold')

    h_list = sorted(all_results.keys())
    f1s = [all_results[h]['test_metrics']['f1'] for h in h_list]
    recalls = [all_results[h]['test_metrics']['recall'] for h in h_list]
    precisions = [all_results[h]['test_metrics']['precision'] for h in h_list]
    aucs = [all_results[h]['test_metrics']['auc_pr'] for h in h_list]

    axes[0].plot(h_list, f1s, 'o-', color='#534AB7', lw=2, label='F1')
    axes[0].plot(h_list, recalls, 's-', color='#D85A30', lw=2, label='Recall')
    axes[0].plot(h_list, precisions, '^-', color='#1D9E75', lw=2, label='Precision')
    axes[0].set_xlabel('Prediction Horizon (ms)')
    axes[0].set_ylabel('Score')
    axes[0].set_title('F1 / Precision / Recall')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xticks(h_list)

    axes[1].plot(h_list, aucs, 'o-', color='#534AB7', lw=2)
    axes[1].set_xlabel('Prediction Horizon (ms)')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('Area Under PR Curve')
    axes[1].grid(True, alpha=0.2)
    axes[1].set_xticks(h_list)

    plt.tight_layout()
    out_dir = Path(kwargs.get('output_dir', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'horizon_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  HORIZON COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Horizon':>8s} {'F1':>8s} {'Prec':>8s} {'Recall':>8s} {'AUC-PR':>8s}")
    print(f"  {'-'*44}")
    for h in h_list:
        m = all_results[h]['test_metrics']
        print(f"  {h:>6d}ms {m['f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['auc_pr']:>8.4f}")

    # Save combined results
    with open(out_dir / 'all_horizons.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    return all_results


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='BE124 GRU Training')
    parser.add_argument('--data-dir', default='dataset_100hz/', help='Dataset directory')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    parser.add_argument('--horizons', nargs='+', type=int, default=[500],
                        help='Prediction horizons to train (default: 500)')
    parser.add_argument('--hidden', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()

    if len(args.horizons) > 1:
        run_all_horizons(
            args.data_dir, args.horizons,
            hidden_size=args.hidden, num_layers=args.layers,
            dropout=args.dropout, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
            output_dir=args.output_dir
        )
    else:
        train_model(
            args.data_dir, horizon_ms=args.horizons[0],
            hidden_size=args.hidden, num_layers=args.layers,
            dropout=args.dropout, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()