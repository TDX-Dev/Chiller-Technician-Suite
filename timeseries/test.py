import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.concat([
    pd.read_csv("../Dataset/ton_efficiency/efficiency/TableData (8).csv"),
])

data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# Fit an ARIMA model
model = ARIMA(data['CH Load'], order=(5, 1, 0))  # Adjust the order as needed
model_fit = model.fit()

# Create a future date range for the next year, hourly
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), 
                              end=data.index[-1] + pd.DateOffset(years=1), 
                              freq='H')

# Generate predictions for the future dates
future_predictions = model_fit.get_forecast(steps=len(future_dates))
predicted_load = future_predictions.predicted_mean

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predicted_load, index=future_dates, columns=['Predicted CH Load'])

# Visualize the predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['CH Load'], label='Historical CH Load', color='blue')
plt.plot(predictions_df.index, predictions_df['Predicted CH Load'], label='Predicted CH Load', color='orange')
plt.title('Chiller Load Prediction for Next Year')
plt.xlabel('Date')
plt.ylabel('CH Load (%)')
plt.legend()
plt.show()

# Save predictions to a CSV file
predictions_df.to_csv('chiller_load_predictions.csv')
