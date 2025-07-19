import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import pickle

def predictionMain(symbol, seq_length=500, forecast_days=15):
    path = f"data/{symbol}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find {path}")
    
    df = pd.read_csv(path)
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - seq_length:]

    X_train, y_train = createSequences(train_data, seq_length)
    X_test, y_test = createSequences(test_data, seq_length)

    model = buildModel(seq_length)
    print("\nTraining model... (this may take a few minutes)")
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1, callbacks=[early_stop])
    model.save(f"models/{symbol}_lstm_model.keras")
    with open(f"models/{symbol}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    visualizeModel(y_test_rescaled, predictions, f"{symbol} - Test Prediction")
    forecastFuture(model, scaled_data, scaler, seq_length, forecast_days, symbol)

def createSequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def buildModel(seq_length):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1),
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecastFuture(model, scaled_data, scaler, seq_length, forecast_days, symbol):
    input_seq = scaled_data[-seq_length:]
    input_seq = input_seq.reshape(1, seq_length, 1)
    future_predictions = []

    print(f"\nForecasting {forecast_days} future days...")

    for _ in range(forecast_days):
        prediction = model.predict(input_seq, verbose=0)[0, 0]
        future_predictions.append(prediction)
        input_seq = np.append(input_seq[:, 1:, :], [[[prediction]]], axis=1)

    predicted_future = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

    plt.figure(figsize=(10, 5))
    plt.plot(predicted_future, label="Predicted Future", color='orange')
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
    predictionMain("AAPL", seq_length=500, forecast_days=15)