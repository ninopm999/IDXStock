# IDX High Dividend 20 Stock Price Predictor with Technical Indicators & Sentiment

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import yfinance as yf

st.set_page_config(page_title="IDX Stock Predictor + Indicators + Sentiment", layout="wide")
st.title("📈 IDX High Dividend 20 – Smart Stock Predictor")

# --- Sidebar Inputs ---
st.sidebar.header("Upload or Select Your Stock Data")
user_file = st.sidebar.file_uploader("Upload CSV with Date, Open, High, Low, Close, Volume", type=['csv'])
selected_symbol = st.sidebar.text_input("Or enter IDX stock symbol (e.g. BBCA.JK)", value="ADRO.JK")

# --- Data Load Function ---
def load_data(symbol):
    data = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

# --- Feature Engineering ---
def add_indicators(data):
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    elif isinstance(close_series.values, np.ndarray) and close_series.values.ndim == 2:
        close_series = close_series.values.squeeze()
    data['RSI'] = RSIIndicator(close=close_series).rsi()
    data['MACD'] = MACD(close=close_series).macd()
    bb = BollingerBands(close=close_series)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data.dropna()

# --- Train Model ---
def train_model(data):
    features = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Day', 'Month', 'Year']
    X = data[features]
    y = data['Close']
    if y.ndim > 1:
        y = y.squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return model, r2, mae, X_test, y_test.reset_index(drop=True), pd.Series(predictions)

# --- Load & Process Data ---
if user_file:
    df = pd.read_csv(user_file)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    df = load_data(selected_symbol)
    st.info(f"Loaded data for: {selected_symbol}")

st.write("✅ Data Preview:", df.tail())
df = add_indicators(df)

# --- Train & Display Results ---
model, r2, mae, X_test, y_test, predictions = train_model(df)
st.success("Model trained successfully!")
st.metric("Model R² Accuracy", f"{r2*100:.2f}%")
st.metric("Mean Absolute Error", f"{mae:.2f} IDR")

# --- Chart Actual vs Predicted ---
result_df = pd.DataFrame({
    'Actual': y_test.values.squeeze(),
    'Predicted': predictions.values.squeeze()
})
st.line_chart(result_df)

# --- Predict Future ---
st.subheader("📅 Predict Future Price")
days_ahead = st.slider("Days into the future", 1, 30, 5)
future_date = df['Date'].iloc[-1] + timedelta(days=days_ahead)
last_row = df.iloc[-1]
future_features = pd.DataFrame([{
    'Open': last_row['Open'],
    'High': last_row['High'],
    'Low': last_row['Low'],
    'Volume': last_row['Volume'],
    'RSI': last_row['RSI'],
    'MACD': last_row['MACD'],
    'BB_High': last_row['BB_High'],
    'BB_Low': last_row['BB_Low'],
    'Day': future_date.day,
    'Month': future_date.month,
    'Year': future_date.year
}])

future_price = model.predict(future_features)[0]
st.success(f"📈 Predicted Close Price on {future_date.date()}: Rp {future_price:,.2f}")

# --- (Optional) Sentiment Note ---
st.caption("📌 Future enhancement: Integrate sentiment from news or social media APIs like NewsAPI or Twitter.")
