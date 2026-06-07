import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

from dataset import build_datasets
from models  import StockLSTM, StockGRU, BiLSTMClassifier, BiGRUClassifier

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models/'
RESULTS_DIR= 'results/'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_model(ModelClass, path, **kwargs):
    model = ModelClass(**kwargs)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def get_predictions_regression(model, dataset):
    loader = DataLoader(dataset, batch_size=256)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(DEVICE)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)

def get_predictions_classifier(model, dataset):
    loader = DataLoader(dataset, batch_size=256)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE)).cpu().squeeze(1)
            preds.append((torch.sigmoid(logits) >= 0.5).float().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ── 1. plot training curves ───────────────────────────────────────────────────
def plot_loss_curves():
    configs = [
        ('lstm_exact',    'LSTM Exact',    'MSE'),
        ('gru_exact',     'GRU Exact',     'MSE'),
        ('lstm_rolling',  'LSTM Rolling',  'MSE'),
        ('gru_rolling',   'GRU Rolling',   'MSE'),
        ('bilstm_turning','BiLSTM Turning','BCE'),
        ('bigru_turning', 'BiGRU Turning', 'BCE'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (name, title, metric) in zip(axes, configs):
        path = f"{MODELS_DIR}{name}_history.json"
        h    = json.load(open(path))
        ax.plot(h['train'], label='Train')
        ax.plot(h['val'],   label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training & Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}loss_curves.png", dpi=150)
    plt.close()
    print("Saved: results/loss_curves.png")


# ── 2. test MSE per model and per d-horizon ───────────────────────────────────
def evaluate_regression():
    _, _, test_exact   = build_datasets(mode='exact')
    _, _, test_rolling = build_datasets(mode='rolling')

    models = [
        (StockLSTM,  'lstm_exact',   test_exact,   'LSTM Exact'),
        (StockGRU,   'gru_exact',    test_exact,   'GRU Exact'),
        (StockLSTM,  'lstm_rolling', test_rolling, 'LSTM Rolling'),
        (StockGRU,   'gru_rolling',  test_rolling, 'GRU Rolling'),
    ]

    print("\n" + "="*60)
    print("REGRESSION TEST RESULTS")
    print("="*60)

    all_results = {}
    for ModelClass, name, test_ds, label in models:
        model = load_model(ModelClass, f"{MODELS_DIR}{name}_best.pt")
        preds, targets = get_predictions_regression(model, test_ds)

        # overall MSE
        overall_mse = np.mean((preds - targets)**2)

        # per d-horizon MSE (d=1..5)
        per_d_mse = [np.mean((preds[:,d] - targets[:,d])**2) for d in range(5)]

        all_results[label] = {'overall': overall_mse, 'per_d': per_d_mse}

        print(f"\n{label}")
        print(f"  Overall test MSE: {overall_mse:.6f}")
        for d, mse in enumerate(per_d_mse, 1):
            print(f"  d={d} MSE: {mse:.6f}")

    # plot per-d MSE comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    d_labels = [f'd={i}' for i in range(1, 6)]
    x = np.arange(5)
    width = 0.2
    for i, (label, res) in enumerate(all_results.items()):
        ax.bar(x + i*width, res['per_d'], width, label=label)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(d_labels)
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE per Forecast Horizon (d=1..5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}per_d_mse.png", dpi=150)
    plt.close()
    print("\nSaved: results/per_d_mse.png")

    return all_results


# ── 3. predicted vs actual for d=1 ───────────────────────────────────────────
def plot_predictions():
    _, _, test_ds = build_datasets(mode='exact')
    model = load_model(StockLSTM, f"{MODELS_DIR}lstm_exact_best.pt")
    preds, targets = get_predictions_regression(model, test_ds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # d=1 scatter
    ax = axes[0]
    ax.scatter(targets[:,0], preds[:,0], alpha=0.3, s=10)
    lim = max(abs(targets[:,0]).max(), abs(preds[:,0]).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Return (d=1)')
    ax.set_ylabel('Predicted Return (d=1)')
    ax.set_title('LSTM Exact: Predicted vs Actual (d=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # d=1 time series (first 100 samples)
    ax = axes[1]
    ax.plot(targets[:100, 0], label='Actual',    alpha=0.8)
    ax.plot(preds[:100,  0], label='Predicted', alpha=0.8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Return (d=1)')
    ax.set_title('LSTM Exact: Time Series (first 100 test samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}predictions.png", dpi=150)
    plt.close()
    print("Saved: results/predictions.png")


# ── 4. classifier evaluation ──────────────────────────────────────────────────
def evaluate_classifier():
    _, _, test_ds = build_datasets(mode='turning')

    models = [
        (BiLSTMClassifier, 'bilstm_turning', 'BiLSTM'),
        (BiGRUClassifier,  'bigru_turning',  'BiGRU'),
    ]

    print("\n" + "="*60)
    print("CLASSIFIER TEST RESULTS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (ModelClass, name, label) in zip(axes, models):
        model = load_model(ModelClass, f"{MODELS_DIR}{name}_best.pt")
        preds, targets = get_predictions_classifier(model, test_ds)

        print(f"\n{label}")
        print(classification_report(targets, preds,
                                    target_names=['Pass','Buy'],
                                    digits=3))

        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Pass','Buy'])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f'{label} Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved: results/confusion_matrix.png")


# ── 5. exact vs rolling MSE comparison ───────────────────────────────────────
def plot_exact_vs_rolling():
    _, _, test_exact   = build_datasets(mode='exact')
    _, _, test_rolling = build_datasets(mode='rolling')

    lstm_exact   = load_model(StockLSTM, f"{MODELS_DIR}lstm_exact_best.pt")
    lstm_rolling = load_model(StockLSTM, f"{MODELS_DIR}lstm_rolling_best.pt")

    p_e, t_e = get_predictions_regression(lstm_exact,   test_exact)
    p_r, t_r = get_predictions_regression(lstm_rolling, test_rolling)

    mse_exact   = [np.mean((p_e[:,d]-t_e[:,d])**2) for d in range(5)]
    mse_rolling = [np.mean((p_r[:,d]-t_r[:,d])**2) for d in range(5)]

    # training stability: std of val loss across epochs
    h_e = json.load(open(f"{MODELS_DIR}lstm_exact_history.json"))
    h_r = json.load(open(f"{MODELS_DIR}lstm_rolling_history.json"))
    std_exact   = np.std(h_e['val'])
    std_rolling = np.std(h_r['val'])

    print("\n" + "="*60)
    print("EXACT vs ROLLING COMPARISON (LSTM)")
    print("="*60)
    print(f"  Val loss std — Exact: {std_exact:.6f}  Rolling: {std_rolling:.6f}")
    print(f"  {'More stable:':15s} {'Rolling' if std_rolling < std_exact else 'Exact'}")
    for d in range(5):
        print(f"  d={d+1}  Exact MSE={mse_exact[d]:.6f}  "
              f"Rolling MSE={mse_rolling[d]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(h_e['val'], label=f'Exact (std={std_exact:.5f})')
    ax.plot(h_r['val'], label=f'Rolling (std={std_rolling:.5f})')
    ax.set_title('Validation Loss: Exact vs Rolling (LSTM)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(5)
    ax.bar(x - 0.2, mse_exact,   0.4, label='Exact')
    ax.bar(x + 0.2, mse_rolling, 0.4, label='Rolling')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={i+1}' for i in range(5)])
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE per Horizon: Exact vs Rolling')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}exact_vs_rolling.png", dpi=150)
    plt.close()
    print("Saved: results/exact_vs_rolling.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_loss_curves()
    evaluate_regression()
    plot_predictions()
    evaluate_classifier()
    plot_exact_vs_rolling()
    print("\nAll evaluation complete. Check the results/ folder.")