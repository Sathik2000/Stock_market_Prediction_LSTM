#!/usr/bin/env python
# coding: utf-8

# In[287]:


import pandas as pd
import matplotlib.pyplot as plt


# In[288]:


df1=pd.read_csv('MSFT.csv')


# In[289]:


df1.head(5)


# In[290]:


df1.shape


# In[291]:


df1.isnull().sum()


# In[292]:


df2=df1.reset_index()['High']


# In[293]:


df3=df1.reset_index()['Close']


# In[294]:


#High
df2.plot()
plt.grid()


# In[295]:


#Close
df3.plot()
plt.grid()


# In[296]:


plt.plot(df1.Open,label='Open')
plt.plot(df1.Close,label='Close')
plt.legend()
plt.grid()
plt.show()


# In[297]:


plt.plot(df1.High,label='High')
plt.plot(df1.Low,label='Low')
plt.legend()
plt.grid()
plt.show()


# In[298]:


df1


# In[299]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already loaded your data into a DataFrame called df1

# Create a copy of df1 and set the 'Date' column as the index
date_ind = df1[['Date', 'Close', 'Open', 'High']]
date_ind.set_index('Date', inplace=True)

# Create a plot from the copied DataFrame
date_ind.plot()
plt.grid()

plt.show()  # To display the plot


# In[300]:


# Before scalling
df3


# In[301]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df3=scaler.fit_transform(np.array(df3).reshape(-1,1))


# In[302]:


df3


# In[303]:


# splitting Data into traintest split
training_size=int(len(df3)*0.65)
test_size=len(df3)-training_size
train_data,test_data=df3[0:training_size,:],df3[training_size:len(df3),:1]


# In[304]:


print('training_size:\n',training_size)
print('test_size:\n',test_size)
print('train_data\n',train_data)
print('test_data\n',test_data)


# In[305]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[355]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 20
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[356]:


print(X_train.shape), print(y_train.shape)


# In[357]:


print(X_test.shape), print(ytest.shape)


# In[358]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[359]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[360]:


model=Sequential()
model.add(LSTM(10,return_sequences=True,input_shape=(20,1)))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[361]:


model.summary()


# In[363]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=10,verbose=1)


# In[364]:


import tensorflow as tf
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[365]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[366]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[367]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[369]:


### Plotting 
# shift train predictions for plotting
look_back=20
trainPredictPlot = numpy.empty_like(df3)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df3)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df3)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df3),label='Complete_Data')
plt.plot(trainPredictPlot,label='P_Training_data')
plt.plot(testPredictPlot,label='P_Test_Data')
plt.legend()
plt.show()


# In[370]:


len(test_data)


# In[371]:


x_input=test_data[68:].reshape(1,-1)
x_input.shape


# In[372]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[373]:


temp_input


# In[374]:


n_steps=20
i=0
while(i<30):
    
    if(len(temp_input)>20):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[375]:


day_new=np.arange(1,21)
day_pred=np.arange(21,51)


# In[376]:


len(df3)


# In[377]:


# Assuming you have defined df3 and lst_output previously

# Convert df3 to a list
df4 = df3.tolist()
df4.extend(lst_output)
combined_array = np.array(df4)

plt.plot(combined_array[100:])
plt.show()


# In[378]:


df4=scaler.inverse_transform(df4).tolist()
plt.plot(df4)


# In[ ]:




