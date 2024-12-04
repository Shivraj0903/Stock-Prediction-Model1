import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2024-03-31'

st.title('Stock Market Prediction')

# Take user input for the stock ticker
user_input = st.text_input('Enter Stock Ticker', value='AAPL')  # Default to AAPL if no input

# Validate if user_input is not empty
if user_input:
    try:
        # Download stock data
        df = yf.download(user_input, start=start, end=end)

        # Check if the DataFrame is not empty
        if not df.empty:
            st.subheader('Data from 2010 - 2024')
            st.write(df.describe())
        else:
            st.error('No data found for the given ticker. Please try another.')
    except Exception as e:
        st.error(f'An error occurred: {e}')
else:
    st.warning('Please enter a stock ticker to proceed.')

# Plotting the closing price vs time chart
if not df.empty:
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.title(f'Closing Price of {user_input} over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    st.pyplot(fig)

    # Plotting Closing Price vs Time with 100-day moving average
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100 Day Moving Average', color='red')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title(f'Closing Price and 100 Day Moving Average for {user_input}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend(loc='best')
    st.pyplot(fig)

    # Plotting Closing Price vs Time with 100MA & 200MA
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100 Day MA', color='red')
    plt.plot(ma200, label='200 Day MA', color='green')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title(f'Closing Price with 100MA & 200MA for {user_input}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend(loc='best')
    st.pyplot(fig)
else:
    st.warning('Please enter a valid ticker to display charts.')

    
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame (df['Close'] [int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler (feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1 / scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
st.pyplot(fig2)
