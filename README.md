# Maintenance Prediction using ARIMA and LSTM

This project demonstrates predictive maintenance using time series forecasting methods: **ARIMA** and **LSTM (Long Short-Term Memory)**. The goal is to anticipate potential equipment failure or need for maintenance based on sensor and vibration data.

## 🔧 Problem Statement

In industrial systems, unexpected equipment failure can lead to costly downtimes. Predictive maintenance uses sensor data to forecast future conditions and schedule maintenance in advance.

This project uses:
- **ARIMA**: A classical statistical approach for univariate time series forecasting.
- **LSTM**: A deep learning-based sequence model ideal for learning long-term dependencies.

---

## 📁 Project Structure

<img width="235" height="334" alt="image" src="https://github.com/user-attachments/assets/9428e9d3-fecc-428d-ba65-fa994c28d13d" />


---

## 📊 Data Used

- **Sensor Data**: Contains time-series readings (like temperature, pressure, voltage) used in ARIMA modeling.
- **Vibration Data**: Time-series vibration readings from machines, suitable for LSTM training.

> Both datasets simulate real-world maintenance conditions in an industrial environment.

---

## 🧠 Models Used

### 🔹 ARIMA (AutoRegressive Integrated Moving Average)
- Forecasts future values based on past sensor readings.
- Used for **short-term prediction** and trend analysis.

### 🔹 LSTM (Long Short-Term Memory)
- A Recurrent Neural Network model tailored for sequence data.
- Captures long-term dependencies in vibration signals.
- More effective in complex, nonlinear patterns.

---

## 🚀 Getting Started

### Install Dependencies

pip install pandas numpy matplotlib seaborn tensorflow statsmodels scikit-learn
RUN :
python arima_prediction.py
LSTM RUN:
Best on Colab(If long path not enabled)
python lstm_prediction.py


