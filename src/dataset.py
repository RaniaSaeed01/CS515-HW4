import yfinance as yf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

# ── constants ────────────────────────────────────────────────────────────────
TICKERS   = ['AAPL', 'MSFT', 'GOOGL']
FEATURES  = ['Open', 'High', 'Low', 'Close']
T         = 20
D         = 5
GAMMA     = 0.02   # 2% gain threshold — realistic for 5-day large-cap returns   

TRAIN_END  = '2024-07-31'
VAL_START  = '2024-08-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END   = '2025-12-31'


# ── 1. download ───────────────────────────────────────────────────────────────
def download_data(tickers=TICKERS, start='2020-01-01', end='2025-12-31',
                  save_dir='data/'):
    all_data = {}
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df = df[FEATURES].dropna()
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df.to_csv(f"{save_dir}{ticker}.csv")
        all_data[ticker] = df
        print(f"  {ticker}: {len(df)} trading days")
    return all_data


# ── 2. load ───────────────────────────────────────────────────────────────────
def load_data(tickers=TICKERS, data_dir='data/'):
    all_data = {}
    for ticker in tickers:
        df = pd.read_csv(
            f"{data_dir}{ticker}.csv",
            index_col=0,
            parse_dates=True,
            date_format='%Y-%m-%d'
        )
        # drop any junk header rows yfinance injects
        df = df[~df.index.astype(str).str.contains('Ticker|Price', na=False)]
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=FEATURES)
        all_data[ticker] = df
    return all_data


# ── 3. split ──────────────────────────────────────────────────────────────────
def split_data(df):
    train = df[df.index <= TRAIN_END]
    val   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df[df.index >= TEST_START]
    return train, val, test


# ── 4. normalise features only (NOT prices used for targets) ──────────────────
def get_scaler_stats(train_df):
    mean = train_df[FEATURES].mean()
    std  = train_df[FEATURES].std().replace(0, 1)
    return mean, std

def normalise(df, mean, std):
    df_norm = df.copy()
    df_norm[FEATURES] = (df[FEATURES] - mean) / std
    return df_norm


# ── 5. target helpers (always use RAW prices) ─────────────────────────────────
def make_return_targets(raw_closes, t, D=D):
    """Exact d-day return ratios."""
    p_t = raw_closes[t]
    return np.array([(raw_closes[t+d] - p_t) / p_t for d in range(1, D+1)],
                    dtype=np.float32)

def make_rolling_targets(raw_closes, t, D=D, l=3):
    """Weighted rolling-average return ratios (uniform weights)."""
    p_t = raw_closes[t]
    w   = np.ones(l+1) / (l+1)
    targets = []
    for d in range(1, D+1):
        if t + d - l < 0:
            targets.append(np.nan)
            continue
        avg = sum(w[j] * raw_closes[t+d-j] for j in range(l+1))
        targets.append((avg - p_t) / p_t)
    return np.array(targets, dtype=np.float32)

def make_turning_label(raw_highs, raw_closes, t, D=D, gamma=GAMMA):
    """
    Buy signal if price ratio (p_max / p_t) > 1 + gamma for any d in 1..D
    gamma=0.1 means a 10% gain threshold
    """
    p_t = raw_closes[t]
    for d in range(1, D+1):
        price_ratio = raw_highs[t+d] / p_t
        if price_ratio > (1 + gamma):
            return 1
    return 0


# ── 6. dataset classes ────────────────────────────────────────────────────────
class StockReturnDataset(Dataset):
    def __init__(self, norm_dfs, raw_dfs, mode='exact', T=T, D=D):
        self.samples = []
        for norm_df, raw_df in zip(norm_dfs, raw_dfs):
            raw_closes = raw_df['Close'].values.astype(np.float32)
            norm_feats = norm_df[FEATURES].values.astype(np.float32)
            n = len(norm_feats)
            for t in range(T, n - D):
                window = norm_feats[t-T:t]              # normalised features
                if mode == 'exact':
                    targets = make_return_targets(raw_closes, t, D)
                else:
                    targets = make_rolling_targets(raw_closes, t, D)
                    if np.any(np.isnan(targets)):
                        continue
                self.samples.append((
                    torch.tensor(window,   dtype=torch.float32),
                    torch.tensor(targets,  dtype=torch.float32),
                ))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class TurningPointDataset(Dataset):
    def __init__(self, norm_dfs, raw_dfs, T=T, D=D, gamma=GAMMA):
        self.samples = []
        for norm_df, raw_df in zip(norm_dfs, raw_dfs):
            raw_closes = raw_df['Close'].values.astype(np.float32)
            raw_highs  = raw_df['High'].values.astype(np.float32)
            norm_feats = norm_df[FEATURES].values.astype(np.float32)
            n = len(norm_feats)
            for t in range(T, n - D):
                window = norm_feats[t-T:t]
                label  = make_turning_label(raw_highs, raw_closes, t, D, gamma)
                self.samples.append((
                    torch.tensor(window, dtype=torch.float32),
                    torch.tensor(label,  dtype=torch.float32),
                ))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ── 7. build all splits ───────────────────────────────────────────────────────
def build_datasets(mode='exact', tickers=TICKERS, data_dir='data/'):
    all_data = load_data(tickers, data_dir)

    norm_train, norm_val, norm_test = [], [], []
    raw_train,  raw_val,  raw_test  = [], [], []

    for ticker in tickers:
        df = all_data[ticker]
        tr, va, te = split_data(df)
        mean, std  = get_scaler_stats(tr)

        raw_train.append(tr.copy())
        raw_val.append(va.copy())
        raw_test.append(te.copy())

        norm_train.append(normalise(tr, mean, std))
        norm_val.append(normalise(va,   mean, std))
        norm_test.append(normalise(te,  mean, std))

    if mode == 'turning':
        Cls = TurningPointDataset
        return (Cls(norm_train, raw_train),
                Cls(norm_val,   raw_val),
                Cls(norm_test,  raw_test))
    else:
        return (StockReturnDataset(norm_train, raw_train, mode=mode),
                StockReturnDataset(norm_val,   raw_val,   mode=mode),
                StockReturnDataset(norm_test,  raw_test,  mode=mode))


# ── smoke test ────────────────────────────────────────────────────────────────
        
if __name__ == '__main__':
    all_data = load_data()
    for ticker, df in all_data.items():
        tr, va, te = split_data(df)
        mean, std = get_scaler_stats(tr)
        raw_closes = tr['Close'].values.astype(np.float32)
        # check a few return targets
        targets = make_return_targets(raw_closes, 20, D=5)
        print(f"{ticker} raw close[20]={raw_closes[20]:.2f}  targets={targets}")
        # check normalised features
        norm = normalise(tr, mean, std)
        print(f"{ticker} norm features[20]={norm[FEATURES].values[20]}")        