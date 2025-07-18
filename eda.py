import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

    #Simple Moving Average
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    plt.plot(df["Close"], label="Close Price", color="blue")
    plt.plot(df["SMA50"], label="SMA 50", color="orange")   
    plt.plot(df["SMA200"], label="SMA 200", color="green")  

    plt.title(f"{symbol} - Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def showNullData(df,symbol):
    sns.heatmap(df.isnull(), cbar=False)
    plt.title(f"{symbol} - Missing Data Map")
    plt.show()

if __name__ == "__main__":
    symbols = ["AAPL", "TSLA", "BTC-USD", "ETH-USD"]
    for symbol in symbols:
        df = loadAndClean(symbol)
        #showNullData(df, symbol)
        plotClose(df, symbol)
        print(df.describe())
