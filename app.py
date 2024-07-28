import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from datetime import datetime

# Title of the Streamlit app
st.title("Stock Price Forecasting")

# Inputs for the stock ticker, start date, end date, and forecast horizon
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())
forecast_horizon = st.number_input("Forecast Horizon (days):", min_value=1, value=7)

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
    decomposition = seasonal_decompose(target, model='multiplicative')
    st.subheader("Seasonal Decomposition")
    st.write("Trend, Seasonal, and Residuals")
    st.pyplot(decomposition.plot())

    # Check if data has a seasonal component
    if decomposition.seasonal.std() > 0:
        st.write("Seasonality detected. Using SARIMA model.")
        model = auto_arima(target, seasonal=True, m=12, stepwise=True)
    else:
        st.write("No seasonality detected. Using ARIMA model.")
        model = auto_arima(target, seasonal=False, stepwise=True)

    # Forecasting
    forecast = model.predict(n_periods=forecast_horizon)
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_horizon + 1, closed='right')

    # Plotting the results
    st.subheader("Forecast vs Actuals")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, target, label='Actuals')
    plt.plot(forecast_index, forecast, label='Forecast', linestyle='--')
    plt.legend()
    st.pyplot(plt)

    # Display the forecasted values
    st.write("Forecasted Values")
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    st.dataframe(forecast_df)

