import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

def create_dataset(dataset, time_step=1):
    """
    Creates a dataset for the LSTM model.

    Args:
        dataset (np.array): The input data.
        time_step (int): The number of previous time steps to use as input variables
                         to predict the next time step.

    Returns:
        tuple: A tuple containing the input data (X) and the target data (y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def run_stock_predictor_app():
    """
    The main function to run the Streamlit stock predictor app.
    """
    st.set_page_config(page_title="IDX Stock Price Predictor", layout="wide")

    st.title("IDX Stock Price Predictor")
    st.markdown("""
    This application uses a Long Short-Term Memory (LSTM) neural network to predict the stock price of a chosen company
    on the Indonesia Stock Exchange (IDX).
    """)

    # --- User Input ---
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., BBCA.JK):", "BBCA.JK").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    predict_button = st.sidebar.button("Predict Stock Price")

    if predict_button:
        with st.spinner(f"Fetching data and training model for {ticker}..."):
            try:
                # --- Data Fetching ---
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    st.error("No data found for the given ticker. Please check the ticker symbol.")
                    return

                st.subheader(f"Historical Data for {ticker}")
                st.write(data)

                # --- Data Preprocessing ---
                close_prices = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(close_prices)

                training_size = int(len(scaled_prices) * 0.8)
                test_size = len(scaled_prices) - training_size
                train_data, test_data = scaled_prices[0:training_size, :], scaled_prices[training_size:len(scaled_prices), :1]

                time_step = 100
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # --- LSTM Model ---
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

                # --- Future Predictions ---
                temp_input = list(test_data[len(test_data) - time_step:].flatten())
                lst_output = []
                n_steps = time_step
                i = 0
                while i < 30:  # Predict for the next 30 days
                    if len(temp_input) > 100:
                        x_input = np.array(temp_input[1:])
                        x_input = x_input.reshape(1, -1)
                        x_input = x_input.reshape((1, n_steps, 1))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.extend(yhat[0].tolist())
                        temp_input = temp_input[1:]
                        lst_output.extend(yhat.tolist())
                        i += 1
                    else:
                        x_input = np.array(temp_input).reshape((1, n_steps, 1))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.extend(yhat[0].tolist())
                        lst_output.extend(yhat.tolist())
                        i += 1

                future_predictions = scaler.inverse_transform(lst_output)

                # --- Visualization ---
                st.subheader("Stock Price Prediction")

                fig = go.Figure()

                # Original Data
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Original Close Price'))

                # Training Predictions
                train_predict_plot = np.empty_like(scaled_prices)
                train_predict_plot[:, :] = np.nan
                train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
                fig.add_trace(go.Scatter(x=data.index, y=train_predict_plot.flatten(), mode='lines', name='Training Predictions'))

                # Test Predictions
                test_predict_plot = np.empty_like(scaled_prices)
                test_predict_plot[:, :] = np.nan
                test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_prices) - 1, :] = test_predict
                fig.add_trace(go.Scatter(x=data.index, y=test_predict_plot.flatten(), mode='lines', name='Test Predictions'))
                
                # Future Predictions
                future_dates = pd.to_datetime(data.index[-1]) + pd.to_timedelta(np.arange(1, 31), 'D')
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Predictions (30 Days)'))


                fig.update_layout(
                    title=f"{ticker} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Stock Price (IDR)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_stock_predictor_app()
