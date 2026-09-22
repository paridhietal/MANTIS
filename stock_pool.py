import torch
import torch.nn.functional as F
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC", "XOM", "CVX", "KO", "PEP", "COST",
    "WMT", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN", "IBM", "ORCL",
]


def build_financial_pretraining_pool(tickers=TICKERS, lookback=60, stride=5,
                                       target_len=512, start="2010-01-01", end="2024-01-01"):
    """
    Downloads daily closes for each ticker, extracts overlapping windows
    (stepping by `stride` days instead of every day, to reduce redundancy),
    z-normalizes each window, resamples to target_len.
    Returns one big unlabeled (N, target_len) tensor for contrastive pretraining.
    """
    all_windows = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            closes = torch.tensor(df["Close"].values, dtype=torch.float32).squeeze()
        except Exception as e:
            print(f"  skipped {ticker}: {e}")
            continue

        if closes.numel() < lookback + 1:
            print(f"  skipped {ticker}: not enough data")
            continue

        windows = []
        for i in range(lookback, len(closes), stride):
            windows.append(closes[i - lookback:i])

        if len(windows) == 0:
            continue

        windows = torch.stack(windows)
        windows = (windows - windows.mean(dim=1, keepdim=True)) / (windows.std(dim=1, keepdim=True) + 1e-8)

        windows = windows.unsqueeze(1)
        windows = F.interpolate(windows, size=target_len, mode="linear", align_corners=False)
        windows = windows.squeeze(1)

        all_windows.append(windows)
        print(f"  {ticker}: {windows.shape[0]} windows")

    pool = torch.cat(all_windows, dim=0)
    return pool


if __name__ == "__main__":
    print("Building financial pretraining pool...")
    pool = build_financial_pretraining_pool()
    print(f"\nTotal financial pretraining windows: {pool.shape}")