import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

def fetch_finnhub_data(ticker, api_key, start_date, end_date):
    """
    Fetches historical stock data from Finnhub API.
    """
    # Convert dates to Unix timestamps
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))

    # Finnhub API URL
    url = 'https://finnhub.io/api/v1/stock/candle'
    
    params = {
        'symbol': ticker,
        'resolution': 'D',
        'from': start_timestamp,
        'to': end_timestamp,
        'token': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # Check if Finnhub returned an empty or error response
        if data.get('s') == 'no_data' or not data.get('c'):
            return None

        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['t'], unit='s')
        df.set_index('Date', inplace=True)
        
        # Rename columns to be compatible with the rest of the app
        df.rename(columns={'c': 'Close', 'h': 'High', 'l': 'Low', 'o': 'Open', 'v': 'Volume'}, inplace=True)
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching data: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing data from Finnhub: {e}")
        return None


def create_dataset(dataset, time_step=1):
    """
    Creates a dataset for the LSTM model.
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

    st.title("🇮🇩 IDX Stock Price Predictor (via Finnhub API)")
    st.markdown("""
    Aplikasi ini menggunakan API resmi dari **Finnhub** dan model *Long Short-Term Memory* (LSTM) 
    untuk memprediksi harga saham di Bursa Efek Indonesia (IDX).
    """)

    # --- User Input ---
    st.sidebar.header("User Input")
    
    st.sidebar.info(
        "**Penting:** Dapatkan API Key gratis Anda dari [Finnhub Dashboard](https://finnhub.io/dashboard) "
        "untuk menggunakan aplikasi ini."
    )
    
    api_key = st.sidebar.text_input("Enter Your Finnhub API Key", type="password", value="YOUR_API_KEY_HERE")
    
    st.sidebar.markdown(
        "**Contoh Kode Saham (harus diakhiri `.JK`):**\n\n"
        "- `BBCA.JK` (BCA)\n"
        "- `TLKM.JK` (Telkom Indonesia)\n"
        "- `GOTO.JK` (GoTo Gojek Tokopedia)\n"
        "- `ASII.JK` (Astra International)"
    )
    
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "BBCA.JK").upper()
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    
    yesterday = date.today() - timedelta(days=1)
    end_date = st.sidebar.date_input("End Date", yesterday)
    
    predict_button = st.sidebar.button("Predict Stock Price")

    if predict_button:
        # --- FIX: Validate that the end date is not in the future ---
        if end_date >= date.today():
            st.error("Tanggal Akhir ('End Date') tidak boleh hari ini atau di masa depan. Silakan pilih tanggal kemarin atau sebelumnya.")
            return

        if api_key == "YOUR_API_KEY_HERE" or not api_key:
            st.error("Harap masukkan API Key Finnhub Anda di sidebar.")
            return

        with st.spinner(f"Fetching data for {ticker} and training model..."):
            
            data = fetch_finnhub_data(ticker, api_key, start_date, end_date)
            
            if data is None or data.empty:
                st.error(f"Tidak ada data yang ditemukan untuk '{ticker}'. Kemungkinan penyebab:\n\n"
                         "1. **API Key tidak valid.**\n"
                         "2. **Kode saham salah.**\n"
                         "3. **Tidak ada data pada rentang tanggal yang dipilih.**\n\n"
                         "Mohon periksa kembali input Anda.")
                return

            st.subheader(f"Historical Data for {ticker}")
            st.write(data)

            # --- Data Preprocessing ---
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(close_prices)

            training_size = int(len(scaled_prices) * 0.8)
            train_data, test_data = scaled_prices[0:training_size, :], scaled_prices[training_size:len(scaled_prices), :1]

            time_step = 100
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            if len(X_train) == 0 or len(X_test) == 0:
                st.warning("Data tidak cukup untuk melatih model. Harap pilih rentang tanggal yang lebih panjang (misalnya, beberapa tahun).")
                return

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # --- LSTM Model ---
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_step, 1)), Dropout(0.2),
                LSTM(50, return_sequences=True), Dropout(0.2),
                LSTM(50), Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=0)

            # --- Predictions ---
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # --- Future Predictions (30 days) ---
            x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
            temp_input = list(x_input[0])
            lst_output = []
            n_steps = time_step
            i = 0
            while(i < 30):
                if(len(temp_input) > time_step):
                    x_input = np.array(temp_input[1:])
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = np.array(temp_input).reshape((1, n_steps,1))
                    yhat = model.predict(x_input,verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i=i+1
            
            future_predictions = scaler.inverse_transform(lst_output)

            # --- Visualization ---
            st.subheader("Hasil Prediksi Harga Saham")
            fig = go.Figure()

            # Plot Actual Price
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Harga Aktual'))

            # Plot Training Predictions
            train_predict_plot = np.empty_like(scaled_prices)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
            fig.add_trace(go.Scatter(x=data.index, y=train_predict_plot.flatten(), mode='lines', name='Prediksi Training'))

            # Plot Test Predictions (More Robust Method)
            test_predict_plot = np.empty_like(scaled_prices)
            test_predict_plot[:, :] = np.nan
            test_start_index = len(train_predict) + (time_step*2) + 1
            test_predict_plot[test_start_index:test_start_index + len(test_predict), :] = test_predict
            fig.add_trace(go.Scatter(x=data.index, y=test_predict_plot.flatten(), mode='lines', name='Prediksi Test'))
            
            # Plot Future Predictions
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Prediksi 30 Hari ke Depan'))

            fig.update_layout(
                title_text=f"Prediksi Harga Saham {ticker}",
                xaxis_title="Tanggal", yaxis_title="Harga Saham (IDR)",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_stock_predictor_app()
