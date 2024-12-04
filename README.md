# Stock-Prediction-Model1

Stock Price Prediction using LSTM
Overview
This project implements a Stock Price Prediction Model using LSTM (Long Short-Term Memory) neural networks. It predicts the future closing price of any company based on historical data obtained from Yahoo Finance.

Table of Contents
Installation
Overview
Features


Installation:
1. download the github repository. make sure all the files are in the same folder.
2. Once done, open the folder and click on address bar and type 'cmd'.(replacing the folder address.
3. Once done, Command prompt will open then type "streamlit run app.py".
4. The web app will open in the browser.
5. enter the stock ticker and it will display the original vs prediction graphs

The project showcases:

Data Retrieval and Preprocessing.

Feature Engineering: Incorporates technical indicators like moving averages.

Model Development: Builds an LSTM-based deep learning model.

Evaluation and Visualization: Compares actual and predicted stock prices.

Features:

Data Scraping: Fetches historical stock data using the yfinance library.

Moving Average Calculation: Computes 100-day and 200-day moving averages for trend analysis.

Data Normalization: Scales data for efficient model training using MinMaxScaler.

LSTM Model:
Handles sequential dependencies in time-series data.
Multiple LSTM layers with dropout to prevent overfitting.

Visualization:
Plots actual vs. predicted prices.



