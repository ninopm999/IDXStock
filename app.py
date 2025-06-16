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

    st.sidebar.header("User Input")

    try:
        stock_df = investpy.get_stocks(country='indonesia')
    except Exception as e:
        st.error(f"Gagal ambil data saham: {e}")
        return

    # Mapping aman langsung dari DataFrame
    stock_display_map = {
        f"{row['name']} ({row['symbol']})": row
        for _, row in stock_df.iterrows()
    }
    stock_display_list = sorted(stock_display_map.keys())

    selected_display = st.sidebar.selectbox("Pilih saham:", stock_display_list)
    selected_stock_row = stock_display_map[selected_display]
    selected_stock_name = selected_stock_row['name']

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    predict_button = st.sidebar.button("Predict Stock Price")

    if predict_button:
        with st.spinner(f"Fetching data for {selected_stock_name}..."):
            try:
                data = investpy.get_stock_historical_data(
                    stock=selected_stock_name,
                    country='indonesia',
                    from_date=start_date.strftime('%d/%m/%Y'),
                    to_date=end_date.strftime('%d/%m/%Y')
                )

                if data.empty:
                    st.error("Data kosong. Coba ubah tanggal atau saham lain.")
                    return

                st.subheader(f"Historical Data: {selected_stock_name}")
                st.write(data)

                close_prices = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(close_prices)

                training_size = int(len(scaled_prices) * 0.8)
                train_data = scaled_prices[:training_size]
                test_data = scaled_prices[training_size:]

                time_step = 100
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                if len(X_train) == 0 or len(X_test) == 0:
                    st.warning("Tidak cukup data. Gunakan rentang tanggal lebih panjang.")
                    return

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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

                train_predict = scaler.inverse_transform(model.predict(X_train))
                test_predict = scaler.inverse_transform(model.predict(X_test))

                x_input = test_data[-time_step:].reshape(1, -1)
                temp_input = list(x_input[0])
                lst_output = []

                for _ in range(30):
                    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.append(yhat[0][0])
                    lst_output.append(yhat[0][0])

                future_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

                st.subheader("Prediction Results")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))

                train_plot = np.empty_like(scaled_prices)
                train_plot[:, :] = np.nan
                train_plot[time_step:len(train_predict)+time_step] = train_predict
                fig.add_trace(go.Scatter(x=data.index, y=train_plot.flatten(), name='Train Prediction'))

                test_index = data.index[len(train_data) + time_step + 1 : len(data) - 1]
                fig.add_trace(go.Scatter(x=test_index, y=test_predict.flatten(), name='Test Prediction'))

                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), name='Future Prediction'))

                fig.update_layout(title=f"{selected_stock_name} Price Forecast", xaxis_title="Date", yaxis_title="IDR", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_stock_predictor_app()
