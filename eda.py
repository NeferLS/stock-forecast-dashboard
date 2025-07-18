import pandas as pd
import matplotlib.pyplot as plt
import os

def loadAndClean(symbol):
    path = f"data/{symbol}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find {path}")
    
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    print(f"\n[{symbol}]") 
    print("Data:")
    print(df.head())
    print("Missing data:")
    print(df.isnull().sum())

    return df

def plotClose(df, symbol):
    row = 10
    col = 4
    plt.figure(figsize=(row,col))
    plt.plot(df["Close"], label=f"{symbol} Close Price")
    plt.title(f"{symbol} - Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbols = ["AAPL", "TSLA", "BTC-USD", "ETH-USD"]
    for symbol in symbols:
        df = loadAndClean(symbol)
        plotClose(df, symbol)
