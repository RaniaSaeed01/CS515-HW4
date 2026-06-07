import sys
sys.path.insert(0, 'src')
from dataset import load_data, split_data

all_data = load_data()
for ticker, df in all_data.items():
    tr, va, te = split_data(df)
    closes = tr['Close'].values.astype(float)
    highs  = tr['High'].values.astype(float)
    count = 0
    total = 0
    for t in range(20, len(closes)-5):
        p_t = closes[t]
        if any(highs[t+d]/p_t > 1.02 for d in range(1,6)):
            count += 1
        total += 1
    print(f'{ticker}: {count}/{total} buy signals ({100*count/total:.1f}%)')