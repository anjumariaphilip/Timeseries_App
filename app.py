import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from datetime import datetime

st.title("Stock Price Forecasting")

ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())
forecast_horizon = st.number_input("Forecast Horizon (days):", min_value=1, value=7)
model_type = st.selectbox("Select Model Type:", ("Holt-Winters", "ARIMA", "SARIMA"))

if st.button("Forecast"):
    # Fetching data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)


    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D')  # Ensure daily frequency
    data['Adj Close'].fillna(method='ffill', inplace=True)  # Fill missing values

    # Select the target variable
    target = data['Adj Close']

decomposition = seasonal_decompose(target, model='multiplicative', period=365)
st.subheader("Seasonal Decomposition")
st.write("Trend, Seasonal, and Residuals")
st.pyplot(decomposition.plot())

monthly_data = target.resample("M").mean()
train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)

st.subheader("Holt-Winters Forecasting")
st.write('Fitting Holt-Winters model...')
hw_model = ExponentialSmoothing(train, seasonal='additive', trend='additive', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(forecast_horizon)
forecast_index = pd.date_range(start=train.index[-1], periods=forecast_horizon + 1, closed='right')

plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast_index, hw_forecast, label='Holt-Winters Forecast', color='green')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Holt-Winters Forecasting')
st.pyplot(plt)
st.write('Holt-Winters Forecasted values:')
hw_forecast_df = pd.DataFrame(hw_forecast, index=forecast_index, columns=['Holt-Winters Forecast'])
st.dataframe(hw_forecast_df)

st.subheader("ARIMA Forecasting")
st.write('Fitting ARIMA model...')
arima_model = auto_arima(train, seasonal=False, suppress_warnings=True)
arima_forecast, arima_conf_int = arima_model.predict(n_periods=forecast_horizon, return_conf_int=True)

plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast_index, arima_forecast, label='ARIMA Forecast', color='blue')
plt.fill_between(forecast_index, 
                     arima_conf_int[:, 0], 
                     arima_conf_int[:, 1], 
                     color='k', alpha=.15)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Forecasting')
st.pyplot(plt)

st.write('ARIMA Forecasted values:')
arima_forecast_df = pd.DataFrame(arima_forecast, index=forecast_index, columns=['ARIMA Forecast'])
st.dataframe(arima_forecast_df)

st.subheader("SARIMA Forecasting")
st.write('Fitting SARIMA model...')
sarima_model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
sarima_forecast, sarima_conf_int = sarima_model.predict(n_periods=forecast_horizon, return_conf_int=True)

plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast_index, sarima_forecast, label='SARIMA Forecast', color='red')
plt.fill_between(forecast_index, 
                 sarima_conf_int[:, 0], 
                 sarima_conf_int[:, 1], 
                 color='k', alpha=.15)


      
forecast_index = pd.date_range(start=target.index[-1], periods=forecast_horizon + 1, closed='right')

    # Plotting the results
st.subheader("Forecast vs Actuals")
plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast_index, forecast, label='Forecast', color='green')
    
    # Plot confidence intervals if available
if conf_int is not None:
   plt.fill_between(forecast_index, 
                    conf_int[:, 0], 
                    conf_int[:, 1], 
                    color='k', alpha=.15)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'{model_type} Forecasting')
st.pyplot(plt)

st.write('SARIMA Forecasted values:')
sarima_forecast_df = pd.DataFrame(sarima_forecast, index=forecast_index, columns=['SARIMA Forecast'])
st.dataframe(sarima_forecast_df)

   
