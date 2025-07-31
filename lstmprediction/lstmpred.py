# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 2: Generate Synthetic Time Series Data
np.random.seed(1)
timesteps = 200
time = np.arange(0, timesteps)
vibration = 0.05 * time + 0.5 * np.sin(0.2 * time) + np.random.normal(0, 0.5, timesteps)

# Let's say equipment failure happens when vibration exceeds a threshold (e.g., 10)
failure_threshold = 10
plt.plot(time, vibration)
plt.axhline(y=failure_threshold, color='r', linestyle='--', label='Failure Threshold')
plt.title("Simulated Equipment Vibration Over Time")
plt.legend()
plt.show()

# Step 3: Prepare Data for LSTM
data = vibration.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(scaled_data, SEQ_LEN)

# Step 4: Build LSTM Model
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Step 5: Train Model
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.1)

# Step 6: Make Predictions
predicted = model.predict(X)
predicted_inverse = scaler.inverse_transform(predicted)
actual_inverse = scaler.inverse_transform(y)

# Step 7: Plot Predictions
plt.plot(actual_inverse, label='Actual Vibration')
plt.plot(predicted_inverse, label='Predicted Vibration')
plt.axhline(y=failure_threshold, color='r', linestyle='--', label='Failure Threshold')
plt.title("LSTM Prediction vs Actual Vibration")
plt.legend()
plt.show()
