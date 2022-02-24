from typing import Text
import yfinance as yf
import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from datetime import date
import pandas_datareader as pdr
from plotly import graph_objs as go
from fbprophet.plot import plot_plotly
import sqlite3 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential 


#Create connection to database (automatically created when called)
connection = sqlite3.connect('database.db')
cursorul = connection.cursor()

#Functions for database with accounts
def create_table():
    cursorul.execute('CREATE TABLE IF NOT EXISTS accounts(username TEXT PRIMARY KEY , password TEXT)')

def add_accounts(username,password):
    cursorul.execute('INSERT INTO accounts(username,password) VALUES (?,?)',(username,password))
    connection.commit()

def login_account(username,password):
    cursorul.execute('SELECT * FROM accounts WHERE username =? and password = ?',(username,password))
    list = cursorul.fetchall()
    return list

def same_name(username):
    cursorul.execute('SELECT username FROM accounts WHERE username =?',(username,))
    list = cursorul.fetchall()
    return list

#Functions for database that stores favorite ticker from users
def create_logbook():
    cursorul.execute('CREATE TABLE IF NOT EXISTS favorites(username TEXT, ticker TEXT)')

def add_tickers(username,ticker):
    cursorul.execute('INSERT INTO favorites(username,ticker) VALUES (?,?)',(username,ticker))
    connection.commit()

def view_favorite_stocks(username):
    cursorul.execute('SELECT ticker FROM favorites WHERE username =?',(username,))
    list = cursorul.fetchall()
    return list

def same_ticker(username,ticker):
    cursorul.execute('SELECT * FROM favorites WHERE username =? and ticker =?',(username,ticker))
    list = cursorul.fetchall()
    return list

def delete_ticker(username,ticker):
    cursorul.execute('DELETE FROM favorites WHERE username =? and ticker=?',(username,ticker))
    connection.commit()

def order_alphabetically(username):
    cursorul.execute('SELECT ticker from favorites WHERE username =? ORDER BY ticker ASC',(username,))
    list = cursorul.fetchall()
    return list 

def order_alphabetically_reverse(username):
    cursorul.execute('SELECT ticker from favorites WHERE username =? ORDER BY ticker DESC',(username,))
    list = cursorul.fetchall()
    return list 

#Find distance between dates
def distance_days(date1, date2):
    return (date2-date1).days

def main():
#Sets the title of the web page
    st.title('Stock Market Web Application - NEA Project')

#Create sidebarheader for user inputs
    st.sidebar.header('Input')
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2021, 11, 20))

#Create a drop-down menu of all the stocks 
    chosen_stock = st.sidebar.selectbox('Choose a stock', ['GOOG', 'MSFT', 'TSLA', 'LCID', 'VLTA', 'AAPL', 'CHPT'])
    stock_data = yf.download(chosen_stock, start_date, end_date)
    stock_data.reset_index(inplace=True)

#Plot graph with opening and closing prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='stock_close'))
    fig.layout.update(title_text="Graph", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

#Let user selected amount of time for prediction
    days = st.selectbox('How many days for the moving average?', [50, 100, 150, 200])

#Predicting closing price
    data_frame = stock_data[['Date','Close']]
    number = distance_days(start_date, end_date)
    if number < 300:
        st.warning("You need to choose a timeframe of at least 300 days for a prediction to be possible")
    else:
        ma = data_frame.Close.rolling(days).mean()
        graph = plt.figure(figsize = (12,6))
        plt.plot(data_frame.Close) 
        plt.plot(ma, 'r')
        st.write(graph)
        
        training = pd.DataFrame(data_frame['Close'][0:int(len(data_frame)*0.70)]) 
        testing = pd.DataFrame(data_frame['Close'][int(len(data_frame)*0.70): int(len(data_frame))])

        scaler = MinMaxScaler(feature_range=(0,1))
        training_array = scaler.fit_transform(training)

        x_train = []
        y_train = []

        for i in range(100, training_array.shape[0]):
            x_train.append(training_array[i-100: i])
            y_train.append(training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

#Machine Learning Model
        model = Sequential()
        model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
                      input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 60, activation = 'relu', return_sequences = True,))
        model.add(Dropout(0.3))

        model.add(LSTM(units = 80, activation = 'relu', return_sequences = True,))
        model.add(Dropout(0.4))

        model.add(LSTM(units = 120, activation = 'relu',))
        model.add(Dropout(0.5))

        model.add(Dense(units = 1))

        model.compile(optimizer='adam', loss = 'mean_squared_error')
        model.fit(x_train, y_train, epochs = 50)

        model.save('keras_model.h5')

        past_days = training.tail(100)
        df = past_days.append(testing, ignore_index = True)
        
        input_data = scaler.fit_transform(df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        
#Make actual prediction
        y_prediction = model.predict(x_test)

#Plot prediction graph
        pre_graph = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'r', label = "Original price")
        plt.plot(y_prediction, 'b', label = "Predicted price")
        plt.xlabel('Time in days')
        plt.ylabel('Performance in Price')
        st.write(testing.shape)
        st.write(pre_graph)
        st.write(y_test)

#Showcases raw data table 
    st.write(stock_data)

#Deal with whitespace (field filled with only spaces or tabs)
def ignore_whitespace(given):
    for n in range(1,len(given)):
        if given[n - 1:n] == given[n:n+1] == " ":
            return given

#Create user registration 
def login():

     st.title("User Login")
     menu = ["Log-in","Sign-up"]
     choose = st.sidebar.selectbox("Menu", menu)
     
     if choose == "Log-in":
        st.subheader("Enter username and password")
        name = st.sidebar.text_input("User-Name")
        password = st.sidebar.text_input("Password",type = 'password')
        if st.sidebar.checkbox("Log-in"):
            
            create_table()
            result = login_account(name, password)
            if result:

                st.success("Logged in as {}".format(name))
                main()

                task = st.selectbox("Options",["Add to favorite tickers","Delete from favorite tickers","Order table alphabetically"])

                #Create favorite stocks section
                if task == "Add to favorite tickers":
                    create_logbook()
                    stock = st.selectbox('Choose a ticker',['GOOG', 'MSFT', 'TSLA', 'LCID', 'VLTA', 'AAPL', 'CHPT'])
                    already_chosen = same_ticker(name, stock)
                    if not st.button("Add ticker"):
                         table = view_favorite_stocks(name)
                         show_table = pd.DataFrame({'ticker':table})
                         st.write(show_table)
                    else:
                         if already_chosen: 
                             st.warning("Ticker has already been choosen")
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame({'ticker':table})
                             st.write(show_table)
                         else:
                             add_tickers(name,stock)
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame({'ticker':table})
                             st.write(show_table)
                elif task == "Delete from favorite tickers":
                    create_logbook()
                    stock = st.selectbox('Choose a ticker',['GOOG', 'MSFT', 'TSLA', 'LCID', 'VLTA', 'AAPL', 'CHPT'])
                    already_chosen = same_ticker(name, stock)
                    if not st.button("Delete ticker"):
                         table = view_favorite_stocks(name)
                         show_table = pd.DataFrame({'ticker':table})
                         st.write(show_table)
                    else:
                         if already_chosen: 
                             delete_ticker(name,stock)
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame({'ticker':table})
                             st.write(show_table)
                         else:
                             st.warning("The selected ticker is not on the list.")
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame({'ticker':table})
                             st.write(show_table)
                elif task == "Order table alphabetically":            
                    create_logbook()
                    if st.button("Order from A to Z", key = "1"):
                        st.button("Order from Z to A", key = "3")
                        alpha_table = order_alphabetically(name)
                        show_alpha_table = pd.DataFrame({'ticker':alpha_table})
                        st.write(show_alpha_table)
                    elif st.button("Order from Z to A", key = "3"):
                        realpha_table = order_alphabetically_reverse(name)
                        show_realpha_table = pd.DataFrame({'ticker':realpha_table})
                        st.write(show_realpha_table)
                    else:
                        table = view_favorite_stocks(name)
                        show_table = pd.DataFrame({'ticker':table})
                        st.write(show_table)

            else:
                st.warning("Incorrect username/password")        
          

     elif choose == "Sign-up":
         st.subheader("Create account")
         name_registered = st.text_input("Username")
         password_registered = st.text_input("Password",type = 'password')

         if st.button("Sign-up"):
             create_table()
             not_unique = same_name(name_registered)
             check_whitespace_name = ignore_whitespace(name_registered)
             check_whitespace_password = ignore_whitespace(password_registered)

             if not_unique or name_registered == '' or password_registered == '':

                 st.warning("Username already exists or one of the text fields is empty")

             elif check_whitespace_name or check_whitespace_password:
                 st.warning("Two or more consecutive empty spaces are not allowed for username or password.")

             else:    
                 add_accounts(name_registered, password_registered)
                 st.success("Account created successfully")

login()
