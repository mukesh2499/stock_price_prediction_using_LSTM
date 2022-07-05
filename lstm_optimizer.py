# here We have tried to change the optimizers for the model and find the accuracy
# You need to import the required packages , the important packages are already been imported in lstm.py
valid_data=final_dataset[987:,:]
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

# fitting the model
lstm_model.compile(loss='mean_squared_error',optimizer='adagrad')
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

