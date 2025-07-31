# predictive_maintenance_arima.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. Generate Synthetic Sensor Data
np.random.seed(42)
healthy_data = np.random.normal(loc=10, scale=2, size=100)
degradation_data = np.arange(10, 50, 0.4) + np.random.normal(loc=0, scale=3, size=100)
sensor_data = pd.Series(np.concatenate([healthy_data, degradation_data]))
sensor_data.index = pd.date_range(start='2025-01-01', periods=200, freq='H')

# 2. Fit ARIMA Model
model = ARIMA(sensor_data, order=(5, 1, 0))
model_fit = model.fit()

# 3. Forecast Next 10 Steps
forecast = model_fit.forecast(steps=10)
forecast_index = pd.date_range(start=sensor_data.index[-1] + pd.Timedelta(hours=1), periods=10, freq='H')

# 4. Plot
plt.figure(figsize=(12, 6))
plt.plot(sensor_data.index, sensor_data, label='Historical Sensor Data')
plt.plot(forecast_index, forecast, label='Forecasted Data (ARIMA)', color='red', marker='o')
plt.axhline(y=55, color='orange', linestyle='--', label='Failure Threshold (55)')
plt.title('ARIMA Model for Predictive Maintenance')
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Failure Alert
if any(forecast > 55):
    print("🚨 ALERT: Predicted values exceed threshold. Maintenance needed!")
else:
    print("✅ System OK. No failure predicted.")
print("Forecast:\n", forecast)
