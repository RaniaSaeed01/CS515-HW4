import torch
import torch.nn as nn


# ── 1. StockLSTM (Part b) ─────────────────────────────────────────────────────
class StockLSTM(nn.Module):
    """
    Stacked LSTM → Dropout → FC
    Input:  (batch, T, F)
    Output: (batch, D)  — D return ratio predictions
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, T, hidden)
        out    = self.dropout(out[:, -1, :])   # take last timestep
        return self.fc(out)             # (batch, D)


# ── 2. StockGRU (Part b) ──────────────────────────────────────────────────────
class StockGRU(nn.Module):
    """
    Stacked GRU → Dropout → FC
    Input:  (batch, T, F)
    Output: (batch, D)
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=5):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ── 3. BiLSTM Classifier (Part d) ────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM → Dropout → FC → Sigmoid
    Input:  (batch, T, F)
    Output: (batch, 1)  — probability of buy signal
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.bilstm(x)                # (batch, T, hidden*2)
        out    = self.dropout(out[:, -1, :])   # last timestep
        return torch.sigmoid(self.fc(out))     # (batch, 1)


# ── 4. BiGRU Classifier (Part d) ─────────────────────────────────────────────
class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU → Dropout → FC → Sigmoid
    Input:  (batch, T, F)
    Output: (batch, 1)
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.bigru = nn.GRU(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.bigru(x)
        out    = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    batch = torch.randn(32, 20, 4)   # 32 samples, T=20, F=4

    for ModelClass, name in [
        (StockLSTM,        'StockLSTM'),
        (StockGRU,         'StockGRU'),
        (BiLSTMClassifier, 'BiLSTMClassifier'),
        (BiGRUClassifier,  'BiGRUClassifier'),
    ]:
        model = ModelClass()
        out   = model(batch)
        print(f"{name:20s}  input={tuple(batch.shape)}  output={tuple(out.shape)}")