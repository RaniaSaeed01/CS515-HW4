import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS   = 'results/'
os.makedirs(RESULTS, exist_ok=True)

M          = 8
SEQ_LEN    = 4
T_ROUNDS   = 4
SIGMA      = 0.5
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 256
BATCH_SIZE = 512
EPOCHS     = 1000
LR         = 1e-3

print(f"Using device: {DEVICE}")


# ── positional encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── transformer block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── TX Encoder ────────────────────────────────────────────────────────────────
class TXEncoder(nn.Module):
    """
    Input per round: for each of 4 symbol positions, concatenate:
      - one-hot of original symbol  (M=8 dims)
      - all received feedback so far (T_ROUNDS dims, zero-padded)
      - round index one-hot          (T_ROUNDS dims)
    → pre-MLP → transformer → post-MLP → power-normalised coded symbol
    """
    def __init__(self):
        super().__init__()
        in_dim = M + T_ROUNDS + T_ROUNDS   # 8 + 4 + 4 = 16
        self.pre_mlp = nn.Sequential(
            nn.Linear(in_dim, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL)
        )
        self.pos_enc = PositionalEncoding(D_MODEL)
        self.blocks  = nn.ModuleList([
            TransformerBlock() for _ in range(N_LAYERS)
        ])
        self.post_mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Linear(D_MODEL // 2, 1)
        )

    def forward(self, msg_onehot, feedback_history, round_idx):
        """
        msg_onehot      : (batch, 4, 8)
        feedback_history: (batch, 4, T_ROUNDS)  — zero-padded for future rounds
        round_idx       : int
        """
        # round one-hot: same for all positions
        round_oh = torch.zeros(msg_onehot.size(0), SEQ_LEN, T_ROUNDS,
                               device=msg_onehot.device)
        round_oh[:, :, round_idx] = 1.0

        z = torch.cat([msg_onehot, feedback_history, round_oh], dim=-1)
        z = self.pre_mlp(z)
        z = self.pos_enc(z)
        for blk in self.blocks:
            z = blk(z)
        x = self.post_mlp(z).squeeze(-1)        # (batch, 4)

        # power normalise: each sample has ||x||^2 / 4 ≤ 1
        scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x     = x / scale                       # unit norm → per-symbol power = 1/4
        return x


# ── RX Decoder ────────────────────────────────────────────────────────────────
class RXDecoder(nn.Module):
    """
    Input: all T received symbols per position → (batch, 4, T)
    Output: logits (batch, 4, 8)
    """
    def __init__(self):
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(T_ROUNDS, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL)
        )
        self.pos_enc = PositionalEncoding(D_MODEL)
        self.blocks  = nn.ModuleList([
            TransformerBlock() for _ in range(N_LAYERS)
        ])
        self.out_mlp = nn.Linear(D_MODEL, M)

    def forward(self, received):
        """received: (batch, T, 4) → permute to (batch, 4, T)"""
        x = received.permute(0, 2, 1)
        x = self.pre_mlp(x)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out_mlp(x)                  # (batch, 4, 8)


# ── full system ───────────────────────────────────────────────────────────────
class CommSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TXEncoder()
        self.decoder = RXDecoder()

    def forward(self, msg, sigma=SIGMA):
        """msg: (batch, 4) integers [0, M-1]"""
        batch = msg.size(0)

        # one-hot encode message: (batch, 4, 8)
        msg_oh = F.one_hot(msg, num_classes=M).float()

        # feedback history buffer: (batch, 4, T_ROUNDS)
        fb_hist = torch.zeros(batch, SEQ_LEN, T_ROUNDS, device=msg.device)
        received = []

        for t in range(T_ROUNDS):
            x_t = self.encoder(msg_oh, fb_hist, round_idx=t)  # (batch, 4)
            y_t = x_t + torch.randn_like(x_t) * sigma         # AWGN
            received.append(y_t)
            fb_hist[:, :, t] = y_t                            # store feedback

        received_stack = torch.stack(received, dim=1)         # (batch, T, 4)
        return self.decoder(received_stack)                   # (batch, 4, 8)


import torch.nn.functional as F


# ── data generator ────────────────────────────────────────────────────────────
def generate_batch(batch_size=BATCH_SIZE, device=DEVICE):
    return torch.randint(0, M, (batch_size, SEQ_LEN), device=device)


# ── training ──────────────────────────────────────────────────────────────────
def train():
    model     = CommSystem().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    history  = {'loss': [], 'ser': []}
    best_ser = 1.0

    print(f"Training for {EPOCHS} epochs | "
          f"T={T_ROUNDS} rounds | sigma={SIGMA} | M={M}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        msg    = generate_batch()
        logits = model(msg)                          # (batch, 4, 8)
        loss   = criterion(logits.reshape(-1, M), msg.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = logits.argmax(dim=-1)
        ser   = (preds != msg).float().mean().item()
        history['loss'].append(loss.item())
        history['ser'].append(ser)

        if ser < best_ser:
            best_ser = ser
            torch.save(model.state_dict(), 'models/comm_best.pt')

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{EPOCHS} | "
                  f"Loss={loss.item():.4f} | SER={ser:.4f}")

    print(f"\nBest training SER: {best_ser:.4f}")
    return model, history


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, n_batches=100):
    model.load_state_dict(torch.load('models/comm_best.pt', map_location=DEVICE))
    model.eval()
    errors, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            msg    = generate_batch()
            logits = model(msg)
            preds  = logits.argmax(dim=-1)
            errors += (preds != msg).sum().item()
            total  += msg.numel()
    ser = errors / total
    print(f"\nTest SER ({n_batches} batches): {ser:.4f}  ({100*ser:.1f}%)")
    print(f"Random baseline SER: {1 - 1/M:.4f}  ({100*(1-1/M):.1f}%)")
    return ser


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['loss'])
    axes[0].set_title('Training Loss (Cross-Entropy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['ser'], label='Model SER')
    axes[1].axhline(y=1 - 1/M, color='r', linestyle='--',
                    label=f'Random baseline ({1-1/M:.3f})')
    axes[1].set_title('Symbol Error Rate During Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SER')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Part 2: Neural Communication Protocol '
                 f'(T={T_ROUNDS}, σ²=0.25)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS}part2_training.png', dpi=150)
    plt.close()
    print("Saved: results/part2_training.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, history = train()
    evaluate(model)
    plot_results(history)