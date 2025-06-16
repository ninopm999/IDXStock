import streamlit as st
import investpy
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
    st.title("🇮🇩 IDX Stock Price Predictor (via investpy)")
    st.markdown("""
    This application uses an LSTM neural network to predict stock prices on the Indonesia Stock Exchange (IDX), powered by data from Investing.com.
    """)

    st.sidebar.header("User Input")

    # Get all available stocks from Indonesia
    try:
        stock_list = investpy.get_stocks(country='indonesia')
    except Exception as e:
        st.error(f"Gagal mengambil daftar saham dari Investing.com: {e}")
        return

    # Display as "Company Name (Symbol)"
    stock_options = [f"{row['name']} ({row['symbol']})" for _, row in stock_list.iterrows()]
    selected_option = st.sidebar.selectbox("Select a stock:", sorted(stock_options))

    # Extract valid name only
    selected_stock = selected_option.split(" (")[0]

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    predict_button = st.sidebar.button("Predict Stock Price")

    if predict_button:
        with st.spinner(f"Fetching data and training model for {selected_stock}..."):
            try:
                data = investpy.get_stock_historical_data(
                    stock=selected_stock,
                    country='indonesia',
                    from_date=start_date.strftime('%d/%m/%Y'),
                    to_date=end_date.strftime('%d/%m/%Y')
                )

                if data.empty:
                    st.error("No data found. Please adjust the date range or try another stock.")
                    return

                st.subheader(f"Historical Data for {selected_stock}")
                st.write(data)

                # --- Preprocessing ---
                close_prices = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(close_prices)

                training_size = int(len(scaled_prices) * 0.8)
                train_data = scaled_prices[0:training_size]
                test_data = scaled_prices[training_size:]

                time_step = 100
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                if len(X_train) == 0 or len(X_test) == 0:
                    st.warning("Not enough data. Expand the date range.")
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
                train_predict = scaler.inverse_transform(model.predict(X_train))
                test_predict = scaler.inverse_transform(model.predict(X_test))

                # --- Future Forecast ---
                x_input = test_data[-time_step:].reshape(1, -1)
                temp_input = list(x_input[0])
                lst_output = []

                for i in range(30):
                    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
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
                train_plot[time_step:len(train_predict)+time_step] = train_predict
                fig.add_trace(go.Scatter(x=data.index, y=train_plot.flatten(), mode='lines', name='Train Prediction'))

                test_index = data.index[len(train_data) + time_step + 1 : len(data) - 1]
                fig.add_trace(go.Scatter(x=test_index, y=test_predict.flatten(), mode='lines', name='Test Prediction'))

                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Prediction'))

                fig.update_layout(title=f"{selected_stock} Price Forecast", xaxis_title="Date", yaxis_title="Price (IDR)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_stock_predictor_app()
