import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from datetime import datetime

st.title("Stock Price Forecasting")

# Inputs for user
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())
forecast_horizon = st.number_input("Forecast Horizon (days):", min_value=1, value=7)

if st.button("Forecast"):
    # Fetching data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Preprocessing
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D')  # Ensure daily frequency
    data['Adj Close'].fillna(method='ffill', inplace=True)  # Fill missing values

    # Select the target variable
    target = data['Adj Close']

    # Seasonal decomposition to check for seasonality
    try:
        decomposition = seasonal_decompose(target, model='additive', period=365)
        st.subheader("Seasonal Decomposition")
        st.write("Trend, Seasonal, and Residuals")
        st.pyplot(decomposition.plot())
        has_seasonality = True
    except ValueError as e:
        st.error(f"Error in seasonal decomposition: {e}")
        has_seasonality = False

    # Resampling the data to monthly frequency for analysis
    monthly_data = target.resample("M").mean()

    # Train-test split
    train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)

    # Fitting ARIMA or SARIMA model based on seasonality
    if has_seasonality:
        st.subheader("SARIMA Forecasting")
        st.write('Fitting SARIMA model...')
        model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_label = 'SARIMA Forecast'
    else:
        st.subheader("ARIMA Forecasting")
        st.write('Fitting ARIMA model...')
        model = auto_arima(train, seasonal=False, suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_label = 'ARIMA Forecast'
      
    # Generate index for forecasted data
    forecast_index = pd.date_range(start=train.index[-1], periods=forecast_horizon + 1, closed='right')

    # Plotting the results
    st.subheader("Forecast vs Actuals")
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Original Data')
    plt.plot(forecast_index, forecast, label=forecast_label, color='green')
    
    # Plot confidence intervals if available
    if conf_int is not None:
        plt.fill_between(forecast_index, 
                         conf_int[:, 0], 
                         conf_int[:, 1], 
                         color='k', alpha=.15)

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{forecast_label}')
    st.pyplot(plt)

    # Display the forecasted values
    st.write('Forecasted values:')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=[forecast_label])
    st.dataframe(forecast_df)
