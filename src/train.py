import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json

from dataset import build_datasets
from models  import StockLSTM, StockGRU, BiLSTMClassifier, BiGRUClassifier

# ── config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
MODELS_DIR = 'models/'
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


# ── training loop (regression) ────────────────────────────────────────────────
def train_regressor(model, train_ds, val_ds, name, epochs=EPOCHS):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5)

    model.to(DEVICE)
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step(val_loss)

        # ── save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODELS_DIR}{name}_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d}/{epochs} | "
                  f"train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

    print(f"[{name}] best val MSE={best_val_loss:.6f}\n")
    json.dump(history, open(f"{MODELS_DIR}{name}_history.json", 'w'))
    return history


# ── training loop (classification) ───────────────────────────────────────────
def train_classifier(model, train_ds, val_ds, name, epochs=EPOCHS):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    labels = torch.tensor([y for _, y in train_ds], dtype=torch.float32)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    pos_weight = (n_neg / n_pos).to(DEVICE)
    print(f"  class balance — neg={int(n_neg)} pos={int(n_pos)} pos_weight={pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # was 1e-3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5)

    model.to(DEVICE)
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_losses = []
        for x, y in train_loader:
            x      = x.to(DEVICE)
            y      = y.to(DEVICE).unsqueeze(1)   # (batch,) → (batch, 1)
            optimizer.zero_grad()
            pred   = model(x)
            loss   = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──
        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x    = x.to(DEVICE)
                y    = y.to(DEVICE).unsqueeze(1)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())
                predicted = (torch.sigmoid(pred) >= 0.5).float()
                correct  += (predicted == y).sum().item()
                total    += y.size(0)

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_acc    = correct / total
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODELS_DIR}{name}_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d}/{epochs} | "
                  f"train BCE={train_loss:.4f} | "
                  f"val BCE={val_loss:.4f} | val acc={val_acc:.3f}")

    print(f"[{name}] best val BCE={best_val_loss:.6f}\n")
    json.dump(history, open(f"{MODELS_DIR}{name}_history.json", 'w'))
    return history


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    '''
    # ── Part b: exact returns ──────────────────────────────────────────────
    print("=" * 60)
    print("PART B — Exact Return Forecasting")
    print("=" * 60)
    train_ds, val_ds, test_ds = build_datasets(mode='exact')

    lstm_exact = StockLSTM()
    train_regressor(lstm_exact, train_ds, val_ds, name='lstm_exact')

    gru_exact = StockGRU()
    train_regressor(gru_exact, train_ds, val_ds, name='gru_exact')

    # ── Part c: rolling average returns ───────────────────────────────────
    print("=" * 60)
    print("PART C — Rolling Average Return Forecasting")
    print("=" * 60)
    train_ds_r, val_ds_r, test_ds_r = build_datasets(mode='rolling')

    lstm_rolling = StockLSTM()
    train_regressor(lstm_rolling, train_ds_r, val_ds_r, name='lstm_rolling')

    gru_rolling = StockGRU()
    train_regressor(gru_rolling, train_ds_r, val_ds_r, name='gru_rolling')
    '''

    # ── Part d: turning point detection ───────────────────────────────────
    print("=" * 60)
    print("PART D — Turning Point Detection")
    print("=" * 60)
    train_ds_t, val_ds_t, test_ds_t = build_datasets(mode='turning')

    bilstm = BiLSTMClassifier()
    train_classifier(bilstm, train_ds_t, val_ds_t, name='bilstm_turning')

    bigru = BiGRUClassifier()
    train_classifier(bigru, train_ds_t, val_ds_t, name='bigru_turning')

    print("All models trained. Checkpoints saved to models/")