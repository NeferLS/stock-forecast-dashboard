import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import pickle

def predictionMain(symbol, seq_length=100, forecast_days=15):
    path = f"data/{symbol}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find {path}")

    df = pd.read_csv(path)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - seq_length:]

    X_train, y_train = createSequences(train_data, seq_length)
    X_test, y_test = createSequences(test_data, seq_length)

    model = buildModel(seq_length, len(features))
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stop])

    model.save(f"models/{symbol}_lstm_model.keras")
    with open(f"models/{symbol}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    predictions = model.predict(X_test, verbose=0)
    dummy_pred = np.zeros((predictions.shape[0], scaler.n_features_in_))
    dummy_pred[:, 3] = predictions[:, 0]
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, 3]

    dummy_y = np.zeros((y_test.shape[0], scaler.n_features_in_))
    dummy_y[:, 3] = y_test
    y_test_rescaled = scaler.inverse_transform(dummy_y)[:, 3]

    visualizeModel(y_test_rescaled, predictions_rescaled, f"{symbol} - Test Prediction")
    forecastFuture(model, scaled_data, scaler, seq_length, forecast_days, symbol)

def createSequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])
    return np.array(X), np.array(y)

def buildModel(seq_length, num_features):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(seq_length, num_features)))
    model.add(Dropout(0.09))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.04))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model

def forecastFuture(model, scaled_data, scaler, seq_length, forecast_days, symbol):
    input_seq = scaled_data[-seq_length:, :]  
    input_seq_model = input_seq[:, 3].reshape(1, seq_length, 1) #using only close col

    future_predictions = []

    print(f"\nForecasting {forecast_days} future days...")

    for _ in range(forecast_days):
        pred = model.predict(input_seq_model, verbose=0)[0, 0]
        future_predictions.append(pred)
        input_seq_model = np.append(input_seq_model[:, 1:, :], [[[pred]]], axis=1)

    dummy_future = np.zeros((forecast_days, scaler.n_features_in_))
    dummy_future[:, 3] = future_predictions
    predicted_future = scaler.inverse_transform(dummy_future)[:, 3]

    dummy_recent = np.zeros((seq_length, scaler.n_features_in_))
    dummy_recent[:, 3] = input_seq[:, 3]
    recent_real = scaler.inverse_transform(dummy_recent)[:, 3]

    plt.figure(figsize=(12, 6))
    plt.plot(range(-seq_length, 0), recent_real, label="Recent Real", color='blue')
    plt.plot(range(0, forecast_days), predicted_future, label="Forecast", color='orange')
    plt.title(f"{symbol} - Forecast for Next {forecast_days} Days")
    plt.xlabel("Future Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualizeModel(realPrices, predictedPrices, title):
    plt.figure(figsize=(12,6))
    plt.plot(realPrices, label='Real Price')
    plt.plot(predictedPrices, label='LSTM Prediction')
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predictionMain("AAPL", seq_length=100, forecast_days=15)