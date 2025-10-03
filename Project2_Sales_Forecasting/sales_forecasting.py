import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Load & prepare
df = pd.read_csv("sales_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.groupby('date')['sales'].sum().reset_index()
df = df.sort_values('date')

# Split train/test (last 30 days for testing)
train = df.iloc[:-30]
test = df.iloc[-30:]

# --- 1. ARIMA Model ---
arima_model = ARIMA(train['sales'], order=(5,1,0))
arima_fit = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=30)
arima_rmse = np.sqrt(mean_squared_error(test['sales'], arima_forecast))
arima_mae = mean_absolute_error(test['sales'], arima_forecast)

# Plot ARIMA
plt.figure(figsize=(10, 6))
plt.plot(train['date'], train['sales'], label='Train')
plt.plot(test['date'], test['sales'], label='Actual')
plt.plot(test['date'], arima_forecast, label='ARIMA Forecast', color='red')
plt.title("ARIMA Sales Forecast")
plt.legend()
plt.show()

# --- 2. Prophet Model ---
prophet_df = train.rename(columns={'date': 'ds', 'sales': 'y'})
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=True)
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

# Prophet RMSE/MAE
prophet_forecast = forecast['yhat'][-30:].values
prophet_rmse = np.sqrt(mean_squared_error(test['sales'], prophet_forecast))
prophet_mae = mean_absolute_error(test['sales'], prophet_forecast)

# Plot Prophet
prophet_model.plot(forecast)
plt.title("Prophet Sales Forecast")
plt.show()

# --- Compare Models ---
print("📊 Model Performance:")
print(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
print(f"Prophet RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")

if arima_rmse < prophet_rmse:
    print("✅ ARIMA performed better!")
else:
    print("✅ Prophet performed better!")

# --- Save forecast results for Power BI ---
test = test.copy()
test['arima_forecast'] = arima_forecast.values
test['prophet_forecast'] = prophet_forecast
test.to_csv("sales_forecast_results.csv", index=False)
print("✅ Forecast results saved to sales_forecast_results.csv")
