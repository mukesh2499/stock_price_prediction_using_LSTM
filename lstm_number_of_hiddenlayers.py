#  You need to import the required packages , the important packages are already been imported in lstm.py
#  Now we will try to increase the number of hidden layers and check the accuracy of the model
valid_data=final_dataset[987:,:]  
lstm_model1=Sequential()
lstm_model1.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model1.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model1.add(LSTM(units=50))
lstm_model1.add(Dense(1))

# fitting the model
lstm_model1.compile(loss='mean_squared_error',optimizer='adam')
lstm_model1.fit(x_train_data,y_train_data,epochs=2,batch_size=1,verbose=2)
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price_model1=lstm_model1.predict(X_test)
closing_price_model1=scaler.inverse_transform(closing_price_model1)
rmse1 = np.sqrt(np.mean(np.power((valid_data-closing_price_model1),2)))
print(rmse1)

train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=closing_price_model1


plt.figure(figsize=(16,8))
plt.title('LSTM Model', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Close Price', fontsize=20)
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
plt.legend(['Train','Val','Predictions'], loc='upper right')
plt.show()

valid_data
