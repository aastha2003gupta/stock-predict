import os
import google.generativeai as genai
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob
import numpy as np
import pandas as pd

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyAT4wXUJkT5yJGL2TglmhTNSjMJSQAhujU")

# Load the pre-trained model for stock prediction
model = load_model('C:/Users/aasth/stock-predict/Stock Predictions Model.keras')

# Supporting functions for volatility and sentiment analysis
def get_stock_volatility(stock_symbol, start_date='2022-01-01', end_date='2024-12-31'):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['Daily Returns'] = stock_data['Close'].pct_change()
    volatility = stock_data['Daily Returns'].std() * np.sqrt(252)  # Annualized volatility
    return volatility, stock_data

def get_stock_sentiment(stock_symbol):
    news_url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey=52790b1a6ad04fcc81429f70a869ea5a'
    news_response = requests.get(news_url).json()
    
    if 'articles' in news_response:
        articles = news_response['articles']
        news_text = " ".join([article['description'] for article in articles if article['description']])
        
        blob = TextBlob(news_text)
        sentiment_score = blob.sentiment.polarity
        return sentiment_score
    else:
        return 0

def assess_risk(volatility, sentiment_score):
    if volatility > 0.03:
        risk = "High"
    elif volatility > 0.01:
        risk = "Medium"
    else:
        risk = "Low"

    if sentiment_score > 0.1:
        recommendation = "Buy"
    elif sentiment_score < -0.1:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return risk, recommendation

# Gemini API Integration for Company Info
def get_company_info(stock_symbol):
    prompt = f"Provide a detailed summary of the company {stock_symbol}, including its industry, market performance, and other relevant details."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Display Company Info Section
def company_info_section(stock_symbol):
    stock_symbol = st.session_state.get('stock_symbol', 'GOOG')  
    st.subheader(f"Company Information for {stock_symbol}")
    try:
        company_info = get_company_info(stock_symbol)
        st.write(company_info)
    except Exception as e:
        st.write("Unable to fetch company info.")
        st.write(str(e))

# Function to display the Hero Page
def hero_page():
    st.title("Welcome to Stock Market Predictor!")
    st.markdown("""
    ## About the App
    This app uses historical stock data and machine learning models to predict future stock prices. 
    It helps you track trends and make informed decisions based on predicted price movements.

    ## Stock Market Overview
    The stock market is a place where investors can buy and sell shares of publicly traded companies. 
    Understanding stock trends is critical for investors looking to maximize their returns.
    """)
    st.image("https://i.etsystatic.com/27344031/r/il/7280ba/3657895905/il_570xN.3657895905_fp3r.jpg", caption="Stock Market Trends")

# Stock Prediction Page
def stock_prediction_page():
    st.header('Stock Market Predictor')

    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    st.session_state['stock_symbol'] = stock
    start = '2022-01-01'
    end = '2024-12-31'

    # Download historical stock data
    data = yf.download(stock, start, end)

    # Display the stock data
    st.subheader('Stock Data')
    st.write(data)

    # Train and test data split
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test)

    # Moving averages plots
    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.title('Price vs MA50')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.title('Price vs MA50 vs MA100')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.title('Price vs MA100 vs MA200')
    plt.show()
    st.pyplot(fig3)

    # Prepare the data for prediction (use the last 100 days for prediction)
    x = []
    y = []
    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i, 0])

    x, y = np.array(x), np.array(y)

    # Model prediction
    predict = model.predict(x)

    # Rescale the prediction and the actual prices
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    # Plot the results (Original vs Predicted)
    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Original vs Predicted Price')
    plt.show()
    st.pyplot(fig4)

    # Predict the next 7 days
    last_100_days = data.Close.tail(100).values
    scaled_last_100_days = scaler.transform(last_100_days.reshape(-1, 1))

    x_input = []
    x_input.append(scaled_last_100_days)
    x_input = np.array(x_input)

    predicted_prices = []
    for i in range(7):  # Predict for 7 days
        prediction = model.predict(x_input)
        predicted_prices.append(prediction[0][0])
        x_input = np.append(x_input[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    predicted_prices = np.array(predicted_prices) * scale

    # Display the predictions for the next week
    st.subheader('Predicted Prices for the Next Week')
    next_week_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
    next_week_prices = pd.DataFrame(predicted_prices, index=next_week_dates, columns=['Predicted Price'])
    st.write(next_week_prices)

    # Plot the predicted prices for the next week
    st.subheader('Predicted Prices for the Next Week')
    fig5 = plt.figure(figsize=(8, 6))
    plt.plot(next_week_prices.index, next_week_prices['Predicted Price'], 'g', label='Predicted Price for Next Week')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Predicted Price for the Next Week')
    plt.show()
    st.pyplot(fig5)

# Risk Analysis Page
def risk_analysis_page():
    st.header('Stock Risk Analysis & Recommendations')

    stock_symbol = st.text_input("Enter Stock Symbol", 'GOOG')

    # Get volatility and stock data
    volatility, stock_data = get_stock_volatility(stock_symbol)

    # Get sentiment from news articles
    sentiment_score = get_stock_sentiment(stock_symbol)

    # Assess risk based on volatility and sentiment
    risk, recommendation = assess_risk(volatility, sentiment_score)

    # Display the results
    st.subheader(f"Risk Assessment for {stock_symbol}")
    st.write(f"Volatility: {volatility:.4f}")
    st.write(f"Sentiment Score: {sentiment_score:.4f}")
    st.write(f"Risk Level: {risk}")
    st.write(f"Recommendation: {recommendation}")

    # Display stock data for reference
    st.subheader("Stock Data")
    st.write(stock_data.tail())

    # Optionally, you can visualize volatility and stock data trends
    st.subheader("Stock Price Trend")
    st.line_chart(stock_data['Close'])

# News Section
# News Section (Updated for finance, business, and stock market news)
def news_section():
    st.subheader("Latest Finance, Business, and Stock Market News")
    # Query for finance, business, or stock market related news
    news_url = f'https://newsapi.org/v2/everything?q=finance OR business OR "stock market"&apiKey=52790b1a6ad04fcc81429f70a869ea5a'
    news_response = requests.get(news_url).json()

    if 'articles' in news_response:
        articles = news_response['articles']
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.write(f"[Read more]({article['url']})")
            st.write("---")
    else:
        st.write("No news found.")


# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("Hero", "Stock Prediction", "Risk Analysis", "News", "Company Info"))

    if page == "Hero":
        hero_page()
    elif page == "Stock Prediction":
        stock_prediction_page()
    elif page == "Risk Analysis":
        risk_analysis_page()
    elif page == "News":
        news_section()  # Example for Google stock
    elif page == "Company Info":
        company_info_section('stock_symbol')  # Example for Google stock

if __name__ == "__main__":
    main()
