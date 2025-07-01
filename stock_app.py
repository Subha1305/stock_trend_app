
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
st.title(" Stock Market Trend Predictor")
st.markdown("This app uses logistic regression to predict whether a stock's closing price will go up or down tomorrow.")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, INFY.NS, RELIANCE.NS):", value="AAPL")

if st.button("Predict"):
    try:
        data = yf.download(stock_symbol, start='2022-01-01', end='2023-12-31')
        if data.empty:
            st.error("No data found. Please check the stock symbol.")
        else:
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            features = data[['Open', 'High', 'Low', 'Volume']].dropna()
            target = data['Target'].dropna()
            features = features.iloc[:-1]
            target = target.iloc[:-1]

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.success(f"Model Accuracy: {accuracy:.2f}")

            latest_data = features.iloc[[-1]]
            prediction = model.predict(latest_data)[0]
            if prediction == 1:
                st.markdown("### Tomorrow's stock price is likely to go ** UP**")
            else:
                st.markdown("###  Tomorrow's stock price is likely to go **DOWN**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
