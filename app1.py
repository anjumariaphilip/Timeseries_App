import streamlit as st
import yfinance as yf
import pandas as pd
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
model_choice = st.selectbox("Choose Forecast Model:", ["ARIMA", "SARIMA"])

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

    # Seasonal decomposition to check for seasonality (only for display purposes)
    try:
        decomposition = seasonal_decompose(target, model='additive', period=30)
        st.subheader("Seasonal Decomposition")
        st.write("Trend, Seasonal, and Residuals")
        st.pyplot(decomposition.plot())
    except ValueError as e:
        st.error(f"Error in seasonal decomposition: {e}")

    # Resampling the data to monthly frequency for analysis
    monthly_data = target.resample("M").mean()
 
    # Train-test split
    train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)

    # Fitting ARIMA or SARIMA model based on user choice
    if model_choice == "SARIMA":
        st.subheader("SARIMA Forecasting")
        st.write('Fitting SARIMA model...')
        model = auto_arima(train, seasonal=True, suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_label = 'SARIMA Forecast'
    else:
        st.subheader("ARIMA Forecasting")
        st.write('Fitting ARIMA model...')
        model = auto_arima(train, seasonal=False, suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        forecast_label = 'ARIMA Forecast'
      
    # Generate index for forecasted data
    forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon)

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

except Exception as e:
        st.error(f"An error occurred: {e}")
   
# Debugging to ensure forecast values are not None or empty
    st.write('Debugging Information:')
    st.write(f'Forecast values: {forecast}')
    if conf_int is not None:
        st.write(f'Confidence intervals: {conf_int}')
