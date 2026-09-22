import torch
import pandas as pd
import yfinance as yf
import torch.nn.functional as F

def load_stock_windows(ticker="AAPL", lookback=60, horizon=5, target_len=512,
                       start="2015-01-01", end="2024-01-01", split_date="2022-01-01"):
    """
    Downloads daily close prices, builds sliding windows of length `lookback`,
    labels each window: 1 if price is higher `horizon` days later, else 0.
    Splits chronologically at `split_date` -- train is everything before, test after.
    """
    split_date = pd.Timestamp(split_date)
    df = yf.download(ticker, start=start, end=end)
    closes = torch.tensor(df["Close"].values, dtype=torch.float32).squeeze()
    dates = df.index
    
    windows, labels, window_dates = [], [], []
    for i in range(lookback, len(closes) - horizon):
        window = closes[i - lookback:i]
        future_price = closes[i + horizon]
        current_price = closes[i]
        label = 1 if  future_price > current_price else 0
        windows.append(window)
        labels.append(label)
        window_dates.append(dates[i])
        
    windows = torch.stack(windows)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # z-normalize each window independently (same preprocessing as training data)
    windows = (windows - windows.mean(dim=1, keepdim=True)) / (windows.std(dim=1, keepdim=True) + 1e-8)
    
    #resample every window to target_len, matching pretraining length
    windows = windows.unsqueeze(1)
    windows = F.interpolate(windows, size=target_len, mode="linear", align_corners=False)
    windows = windows.squeeze(1)
    
    #chronological split
    split_idx = next(i for i, d in enumerate(window_dates) if d >= split_date)
    X_train, y_train = windows[:split_idx], labels[:split_idx]
    X_test, y_test = windows[split_idx:], labels[split_idx:]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_stock_windows()
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("Train label balance:", y_train.float().mean().item())
    print("Test label balance:", y_test.float().mean().item())
        
           