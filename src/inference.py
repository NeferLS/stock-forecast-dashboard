import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from keras.models import load_model

def loadScaler(scaler_path):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

def forecastFuture(model, input_data, scaler, seq_length, forecast_days):
    input_seq = input_data[-seq_length:].reshape(1, seq_length, 1)
    future_predictions = []

    for _ in range(forecast_days):
        prediction = model.predict(input_seq, verbose=0)[0, 0]
        future_predictions.append(prediction)
        input_seq = np.append(input_seq[:, 1:, :], [[[prediction]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return predicted_prices

def plotForecast(predicted_prices, symbol, forecast_days):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_prices, label="Predicted Future")
    plt.title(f"{symbol} - Forecast for Next {forecast_days} Days")
    plt.xlabel("Future Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(symbol, seq_length=500, forecast_days=90):
    model_path = f"models/{symbol}_lstm_model.keras"
    scaler_path = f"models/{symbol}_scaler.pkl"
    data_path = f"data/{symbol}.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    model = load_model(model_path)
    scaler = loadScaler(scaler_path)

    df = pd.read_csv(data_path)
    data = df[['Close']].values
    scaled_data = scaler.transform(data)

    predicted_prices = forecastFuture(model, scaled_data, scaler, seq_length, forecast_days)
    plotForecast(predicted_prices, symbol, forecast_days)

if __name__ == "__main__":
    main("AAPL", seq_length=500, forecast_days=90)