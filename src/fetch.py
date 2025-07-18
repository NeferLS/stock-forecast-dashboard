import yfinance as yf
import os

def download_assets(symbols, start="2018-01-01", end="2024-12-31"):
    os.makedirs("data", exist_ok=True)
    for symbol in symbols:
        print(f"Downloading: {symbol}...")
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            print(f"Error: couldn't fetch data from {symbol}")
            continue
        df = df[["Open", "High","Low", "Close", "Volume"]]
        df.reset_index(inplace=True)
        df.to_csv(f"data/{symbol}.csv", index=False)
        print(f"{symbol} saved in data/{symbol}.csv")

if __name__ == "__main__":
    assets = ["AAPL", "TSLA", "BTC-USD", "ETH-USD"]
    download_assets(assets)