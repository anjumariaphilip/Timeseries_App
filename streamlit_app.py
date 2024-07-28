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

# Title of the Streamlit app
st.title("Stock Price Forecasting")

# Inputs for the stock ticker, start date, end date, forecast horizon, and model type
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())
forecast_horizon = st.number_input("Forecast Horizon (days):", min_value=1, value=7)
model_type = st.selectbox("Select Model Type:", ("Holt-Winters", "ARIMA", "SARIMA"))

# Button to trigger data fetching and model building
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

    # Seasonal decomposition
    decomposition = seasonal_decompose(target, model='multiplicative', period=365)
    st.subheader("Seasonal Decomposition")
    st.write("Trend, Seasonal, and Residuals")
    st.pyplot(decomposition.plot())

    # Train-test split
    train, test = train_test_split(target, test_size=0.2, shuffle=False)

    # Model fitting based on selected model type
    if model_type == "Holt-Winters":
        st.write("Using Holt-Winters method.")
        model = ExponentialSmoothing(train, seasonal='multiplicative', trend='additive', seasonal_periods=365).fit()
        forecast = model.forecast(forecast_horizon)
    elif model_type == "SARIMA":
        st.write("Using SARIMA model.")
        model = auto_arima(train, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
        forecast = model.predict(n_periods=forecast_horizon)
    elif model_type == "ARIMA":
        st.write("Using ARIMA model.")
        model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
        forecast = model.predict(n_periods=forecast_horizon)

    forecast_index = pd.date_range(start=target.index[-1], periods=forecast_horizon + 1, closed='right')

    # Plotting the results
    st.subheader("Forecast vs Actuals")
    plt.figure(figsize=(10, 5))
    plt.plot(target.index, target, label='Actuals')
    plt.plot(forecast_index, forecast, label='Forecast', linestyle='--')
    plt.legend()
    st.pyplot(plt)

    # Display the forecasted values
    st.write("Forecasted Values")
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    st.dataframe(forecast_df)
