import os
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Fetch tickers
try:
    response = requests.get(f"{API_URL}/tickers")
    tickers = response.json().get("tickers", [])
except requests.exceptions.RequestException as e:
    st.error(f"Failed to load tickers: {e}")
    tickers = []

# Streamlit UI
st.title("Stock Price Prediction")
ticker = st.selectbox("Select Ticker", tickers)

# Input fields
open_price = st.number_input("Open Price", min_value=0.0, value=180.50)
high_price = st.number_input("High Price", min_value=0.0, value=182.75)
low_price = st.number_input("Low Price", min_value=0.0, value=179.25)
volume = st.number_input("Volume", min_value=0, value=40000000)
lag_1_close = st.number_input("Previous Close", min_value=0.0, value=180.00)

if st.button("Predict"):
    input_data = {
        "ticker": ticker,
        "data": {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Volume": volume,
            "lag_1_close": lag_1_close
        }
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Close Price: {result['prediction']:.2f}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")