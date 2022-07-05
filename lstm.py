# Code begins here

# Importing the packages
import numpy as np
import pandas as pd
#import streamlit as st


import matplotlib.pyplot as plt

# Setting the size of figure
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

# To Normalize the data 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

# Reading the dataset
df=pd.read_csv("dataf.csv")
# Display the data
df.head(10) # will return the first 10 rows of the dataframe
#st.subheader('Data from 2010-2019')
#st.write(df.describe())
#st.subheader('Closing price vas Time-chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df["Close"])
#st.pyplot(fig)
# Setting the index as date
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

# Visualizing the closing prices by plotting the data 
# Setting the figure size, giving title, labels and their fontsize
plt.figure(figsize=(16,8))
plt.title('Close Price History', fontsize=20)
plt.plot(df["Close"])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Close Price', fontsize=20)
plt.show()

# Importing the libraries as required
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

# Sorting the dataset by date and filtering the “Date” and “Close” columns 
data=df.sort_index(ascending=True,axis=0)

# creating new filtered dataset
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]

    
# Setting the index
new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

# creating the train set and the test set
final_dataset=new_dataset.values
train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]    

# Normalizing the filtered/new dataset
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

# Converting the dataset in the form of x_train_data and y_train_data
x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
# Converting to numpy arrays
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
# Reshaping the data as required by the LSTM model ( It expects 3-Dimensional data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


# we could have used different models and based upon that we could have chose the best one 
# but instead we chose the best one and tried to improve its accuracy



# Building and training the LSTM network model 
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

# fitting the model
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=2,batch_size=1,verbose=2)

# Taking a sample from the dataset to make predictions using the LSTM model

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=lstm_model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

# getting the root mean squared error
rmse = np.sqrt(np.mean(np.power((valid_data-closing_price),2)))
print(rmse)


# Saving the model
lstm_model.save("Stock prediction_LSTM")

# Visualizing the predicted stock costs with the actual stock cost by plotting the data
train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=closing_price


plt.figure(figsize=(16,8))
plt.title('LSTM Model', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Close Price', fontsize=20)
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
plt.legend(['Train','Val','Predictions'], loc='upper right')
plt.show()

valid_data
