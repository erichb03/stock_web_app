from typing import Text
import yfinance as yf
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import date
import pandas_datareader as pdr
from plotly import graph_objs as go
import sqlite3 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow import keras
import requests 
import smtplib
from email.message import EmailMessage

#Email sender function 
def send_email(receiver, the_name, the_password):
    email_name = "streamlitstockmarketapp@gmail.com"
    email_password = "Arnewood98"
    
    message = EmailMessage()
    message['Subject'] = 'Stock Prediction Web App account details:'
    message['From'] = email_name
    message['To'] = receiver

    message.set_content("Account username: {0}\nAccount password: {1}".format(the_name, the_password))
  
  
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        
        smtp.login(email_name, email_password)

        smtp.send_message(message)

#Create connection to database 
connection = sqlite3.connect('sps.db')
cursorul = connection.cursor()

#Functions for database with accounts
def create_table():
    cursorul.execute('CREATE TABLE IF NOT EXISTS account(username TEXT PRIMARY KEY , email TEXT, password TEXT)')
    connection.commit()

def add_accounts(username,email,password):
    cursorul.execute('INSERT INTO account(username,email,password) VALUES (?,?,?)',(username,email,password))
    connection.commit()

def login_account(username,password):
    cursorul.execute('SELECT * FROM account WHERE username =? and password = ?',(username,password))
    list = cursorul.fetchall()
    return list

def same_name(username):
    cursorul.execute('SELECT username FROM account WHERE username =?',(username,))
    list = cursorul.fetchall()
    return list

def same_email(email):
    cursorul.execute('SELECT email FROM account WHERE email =?',(email,))
    list = cursorul.fetchall()
    return list 

def will_delete_later():
    cursorul.execute('SELECT * from account')
    list = cursorul.fetchall()
    return list

def get_password(email):
    cursorul.execute('SELECT password FROM account WHERE email =?',(email,))
    list = cursorul.fetchall()
    for item in list:
        return item[0]

def get_name(email):
    cursorul.execute('SELECT username FROM account WHERE email =?',(email,))
    list = cursorul.fetchall()
    for item in list:
        return item[0]

#Functions for database that stores favorite ticker from users
def create_logbook():
    cursorul.execute('CREATE TABLE IF NOT EXISTS favorites(id INTEGER PRIMARY KEY, username TEXT, ticker TEXT)')

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
    cursorul.execute('SELECT ticker FROM favorites WHERE username =? ORDER BY ticker ASC',(username,))
    list = cursorul.fetchall()
    return list 

def order_alphabetically_reverse(username):
    cursorul.execute('SELECT ticker FROM favorites WHERE username =? ORDER BY ticker DESC',(username,))
    list = cursorul.fetchall()
    return list 

def popular_tickers():
    cursorul.execute('SELECT ticker,count(*) FROM favorites GROUP BY ticker')
    list = cursorul.fetchall()
    return list

#Check if the email actually exists
def check_email(email_address):
    response = requests.get("https://isitarealemail.com/api/email/validate",
                             params = {'email': email_address})
    
    status = response.json()['status']
    
    return status
       
#Find distance between dates
def distance_days(date1, date2):
    return (date2-date1).days

def main():
#Sets the title of the web page
    st.subheader('Stock Market Web Application - NEA Project')
    st.write(' ')
    #test_delete = will_delete_later()
    #table_delete = pd.DataFrame(test_delete)
    #st.dataframe(table_delete)

#Create sidebarheader for user inputs
    st.sidebar.header('Input')
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2021, 11, 20))

#Create a drop-down menu of all the stocks 
    chosen_stock = st.sidebar.selectbox('Choose a stock', ['GOOG', 'MSFT', 'TSLA', 'LCID', 
                                                           'VLTA', 'AAPL', 'CHPT', 'WWE'])
    stock_data = yf.download(chosen_stock, start_date, end_date)
    stock_information = yf.Ticker(chosen_stock)
    stock_data.reset_index(inplace=True)

#Show key information about each ticker
    logo = '<img src=%s>' % stock_information.info['logo_url']
    st.markdown(logo, unsafe_allow_html = True)
    company = stock_information.info['longName']
    st.header('**%s**' % company)
    description = stock_information.info['longBusinessSummary']
    st.info(description)

#Plot graph with opening and closing prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='stock_close'))
    fig.layout.update(title_text="Graph", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

#Let user selected amount of time for prediction
    st.header("Moving average")
    days = st.selectbox('How many days for the moving average?', [50, 100, 150, 200])

    data_frame = stock_data[['Date','Close']]
    number = distance_days(start_date, end_date)
#Check if the number of days is big enough for prediction
    if number < 300:
        st.warning("You need to choose a timeframe of at least 300 days for a prediction to be possible")
    else:
        ma = data_frame.Close.rolling(days).mean()
        graph = plt.figure(figsize = (12,6))
        plt.plot(data_frame.Close) 
        plt.plot(ma, 'r')
        st.write(graph)
        
#Split data into training (70%) and testing (30%)
        training = pd.DataFrame(data_frame['Close'][0:int(len(data_frame)*0.70)]) 
        testing = pd.DataFrame(data_frame['Close'][int(len(data_frame)*0.70): int(len(data_frame))])

        scaler = MinMaxScaler(feature_range=(0,1))

#Load Model
        model = load_model('keras_model.h5')

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
        prediction = model.predict(x_test)
        y_prediction = scaler.inverse_transform(prediction)
        y_test = y_test.reshape(-1,1)
        y_test = scaler.inverse_transform(y_test)
        
#Plot prediction graph
        last_30 = str(y_test.shape[0])
        st.header("Expected value in the last {0} days vs. Actual value in the last {1} days".format(last_30,last_30))
        st.write("This prediction can be used in order to determine if a stock is overperforming, performing as expected or underperforming.")
        pre_graph = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'r', label = "Actual")
        plt.plot(y_prediction, 'b', label = "Expected")
        plt.xlabel('Time in days')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(pre_graph)

#Showcases raw data table 
    st.text("All data from the first 5 days")
    st.write(stock_data.head())
    st.text("All data from the last 5 days")
    st.write(stock_data.tail())

#Deal with whitespace (field filled with only spaces or tabs)
def ignore_whitespace(given):
    for n in range(1,len(given)):
        if given[n - 1:n] == given[n:n+1] == " ":
            return given

#Create user registration 
def login():

     menu = ["Log-in","Sign-up","Restore username and password"]
     choose = st.sidebar.selectbox("Menu", menu)
     
     if choose == "Log-in":
        name = st.sidebar.text_input("User-Name")
        password = st.sidebar.text_input("Password",type = 'password')
        if st.sidebar.checkbox("Log-in"):
            
            create_table()
            result = login_account(name, password)
            if result:

                st.success("Logged in as {}".format(name))
                main()

                task = st.selectbox("Options",["Add to favorite tickers",
                                               "Delete from favorite tickers",
                                               "Order table alphabetically",
                                               "Show most popular tickers"])

                #Create favorite stocks section
                if task == "Add to favorite tickers":
                    create_logbook()
                    stock = st.selectbox('Choose a ticker',['GOOG', 'MSFT', 'TSLA', 'LCID', 
                                                            'VLTA', 'AAPL', 'CHPT', 'WWE'])
                    already_chosen = same_ticker(name, stock)
                    if not st.button("Add ticker"):
                         table = view_favorite_stocks(name)
                         show_table = pd.DataFrame(table, columns = ["Ticker"])
                         st.write(show_table)
                    else:
                         if already_chosen: 
                             st.warning("Ticker has already been choosen")
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame(table, columns = ["Ticker"])
                             st.write(show_table)
                         else:
                             add_tickers(name,stock)
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame(table, columns = ["Ticker"])
                             st.write(show_table)
                elif task == "Delete from favorite tickers":
                    create_logbook()
                    stock = st.selectbox('Choose a ticker',['GOOG', 'MSFT', 'TSLA', 'LCID', 'VLTA', 'AAPL', 'CHPT'])
                    already_chosen = same_ticker(name, stock)
                    if not st.button("Delete ticker"):
                         table = view_favorite_stocks(name)
                         show_table = pd.DataFrame(table, columns = ["Ticker"])
                         st.write(show_table)
                    else:
                         if already_chosen: 
                             delete_ticker(name,stock)
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame(table, columns = ["Ticker"])
                             st.write(show_table)
                         else:
                             st.warning("The selected ticker is not on the list.")
                             table = view_favorite_stocks(name)
                             show_table = pd.DataFrame(table, columns = ["Ticker"])
                             st.write(show_table)
                elif task == "Order table alphabetically":            
                    create_logbook()
                    if st.button("Order from A to Z", key = "1"):
                        st.button("Order from Z to A", key = "3")
                        alpha_table = order_alphabetically(name)
                        show_alpha_table = pd.DataFrame(alpha_table, columns = ['Ticker'])
                        st.write(show_alpha_table)
                    elif st.button("Order from Z to A", key = "3"):
                        realpha_table = order_alphabetically_reverse(name)
                        show_realpha_table = pd.DataFrame(realpha_table, columns = ["Ticker"])
                        st.write(show_realpha_table)
                    else:
                        table = view_favorite_stocks(name)
                        show_table = pd.DataFrame(table, columns = ["Ticker"])
                        st.write(show_table)
                elif task == "Show most popular tickers":
                    create_logbook()
                    popularity = popular_tickers()
                    show_popularity = pd.DataFrame(popularity)
                    st.write(show_popularity)
            else:
                st.warning("Incorrect username/password")        
          

     elif choose == "Sign-up":
         st.subheader("Create account")
         email_registered = st.text_input("Email")
         name_registered = st.text_input("Username")
         password_registered = st.text_input("Password",type = 'password')

         if st.button("Sign-up"):
             create_table()
             not_real = check_email(email_registered)
             not_unique = same_name(name_registered)
             already_used = same_email(email_registered)
             check_whitespace_name = ignore_whitespace(name_registered)
             check_whitespace_password = ignore_whitespace(password_registered)

             if not_unique or name_registered == '' or password_registered == '':

                 st.warning("Username already exists or one of the text fields is empty")

             elif check_whitespace_name or check_whitespace_password:
                 st.warning("Two or more consecutive empty spaces are not allowed for username or password.")

             elif not_real == "invalid" or already_used:
                 st.warning("The email you have entered is not valid or the email has already been used.")

             else:    
                 add_accounts(name_registered, email_registered, password_registered)
                 st.success("Account created successfully")

     elif choose == "Restore username and password":
         st.header("Enter the email used to create your account here:")
         address = st.text_input("Email")
         account_exists = same_email(address)
         if st.button('Restore'):
             if account_exists:
                 name_send = get_name(address)
                 password_send = get_password(address)
                 send_email(address, name_send, password_send)
             else:
                 st.warning('There is no account registered on this email address')
        

login()
