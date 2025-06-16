import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def run_stock_predictor_app():
    st.set_page_config(page_title="IDX Stock Price Predictor", layout="wide")
    st.title("🇮🇩 IDX Stock Price Predictor")
    st.markdown("""
    This application uses a Long Short-Term Memory (LSTM) neural network to predict the stock price of a chosen company
    on the Indonesia Stock Exchange (IDX).
    """)

    # --- User Input ---
    st.sidebar.header("User Input")
    st.sidebar.info("""
    Enter an official IDX ticker symbol. **It must end with `.JK`**.

    **Examples:**
    - `BBCA.JK` (BCA)
    - `TLKM.JK` (Telkom Indonesia)
    - `GOTO.JK` (GoTo Gojek Tokopedia)
    - `ASII.JK` (Astra International)
    """)

    ticker = st.sidebar.text_input("Enter Stock Ticker:", "BBCA.JK").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    predict_button = st.sidebar.button("Predict Stock Price")

    if predict_button:
        if not ticker.endswith('.JK'):
            st.error("Invalid Ticker. It must end with `.JK`.")
            return

        with st.spinner(f"Fetching data and training model for {ticker}..."):
            try:
                # --- Robust Download ---
                data = yf.download(ticker, start=start_date, end=end_date)

                if data.empty:
                    st.error(f"No data found for '{ticker}'. This could be due to:\n\n"
                             "1. Incorrect ticker.\n"
                             "2. Company not listed in that time range.\n"
                             "3. Temporary issue with Yahoo Finance.\n\n"
                             "Please check the ticker or try again later.")
                    return

                st.subheader(f"Historical Data for {ticker}")
                st.write(data)

                # --- Preprocessing ---
                close_prices = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(close_prices)

                training_size = int(len(scaled_prices) * 0.8)
                train_data = scaled_prices[0:training_size, :]
                test_data = scaled_prices[training_size:, :1]

                time_step = 100
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                if len(X_train) == 0 or len(X_test) == 0:
                    st.warning("Not enough data to train. Select a wider date range.")
                    return

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # --- Model ---
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=0)

                # --- Predictions ---
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)

                train_predict = scaler.inverse_transform(train_predict)
                test_predict = scaler.inverse_transform(test_predict)

                # --- Future Prediction ---
                x_input = test_data[-time_step:].reshape(1, -1)
                temp_input = list(x_input[0])
                lst_output = []

                for i in range(30):
                    x_input = np.array(temp_input[-time_step:])
                    x_input = x_input.reshape((1, time_step, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.append(yhat[0][0])
                    lst_output.append(yhat[0][0])

                future_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

                # --- Visualization ---
                st.subheader("Stock Price Prediction Results")
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'))

                train_plot = np.empty_like(scaled_prices)
                train_plot[:, :] = np.nan
                train_plot[time_step:len(train_predict) + time_step, :] = train_predict
                fig.add_trace(go.Scatter(x=data.index, y=train_plot.flatten(), mode='lines', name='Training Prediction'))

                test_start_idx = len(train_data) + time_step + 1
                test_index = data.index[test_start_idx:test_start_idx + len(test_predict)]
                fig.add_trace(go.Scatter(x=test_index, y=test_predict.flatten(), mode='lines', name='Test Prediction'))

                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Prediction (30 Days)'))

                fig.update_layout(
                    title=f"{ticker} Stock Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (IDR)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Try again later or check ticker/date.")

if __name__ == "__main__":
    run_stock_predictor_app()
