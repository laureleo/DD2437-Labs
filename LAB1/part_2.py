# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:26:48 2018

@author: s86852
"""
import tensorflow as ts
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import time
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(9001)

# Parameters for time series
beta = 0.2
gamma = 0.1
n = 10
tau = 25
N = 2000  #lenth of time series

# Initialize array 
x = np.zeros([N])
x[tau] = 1.5

# Generate time series
for i in range(len(x)-tau-1):
    j = i + tau   # j = tau -> t = 0 
    x[j+1] = x[j] + (beta*x[j-tau])/(1 + math.pow(x[j-tau], 10)) - gamma*x[j]

# Show time series
# plt.plot(x)

# Select data for training, validating and testing
t_start = 300; t_end = 1500

input_ts = np.array([x[range(t_start-20, t_end-20)],
                      x[range(t_start-15, t_end-15)],
                      x[range(t_start-10, t_end-10)],
                      x[range(t_start-5, t_end-5)],
                      x[range(t_start, t_end)]])

output_ts = np.array([range(t_start+5, t_end+5)])

n, m = np.shape(input_ts)
t_test = range(m-200, m)
input_test = np.transpose(input_ts[:, t_test])  # transposed the data to comply with keras.. Ugly, I know :)
output_test = np.transpose(output_ts[:, t_test])

num_training = 700
t_train = range(num_training)
input_train = np.transpose(input_ts[:, t_train])
output_train = np.transpose(output_ts[:, t_train])

t_validate = range(num_training, m-200)
input_validate = np.transpose(input_ts[:, t_validate])
output_validate = np.transpose(output_ts[:, t_validate])

# Show training, validating and test data, [NOT WORKING AFTER THE TRANSPOSE]
# plt.plot(t_train, input_train[0,:]); plt.plot(t_validate, input_validate[0,:]); plt.plot(t_test, input_test[0,:])

## Keras ANN ##
# https://keras.io/

# Normalize data
mu_x = np.mean(input_train, axis=0, keepdims=True)
mu_y = np.mean(output_train, axis=0, keepdims=True)

sigma_x = np.std(input_train, axis=0, keepdims=True)
sigma_y = np.std(output_train, axis=0, keepdims=True)

input_train = (input_train-mu_x)/sigma_x
output_train = (output_train-mu_y)/sigma_y

input_validate = (input_validate-mu_x)/sigma_x
output_validate = (output_validate-mu_y)/sigma_y

input_test = (input_test-mu_x)/sigma_x
output_test = (output_test-mu_y)/sigma_y

# Build network
batch_size = 100
epochs = 100

model = Sequential()
model.add(Dense(3, activation='relu', input_dim=5))
for i in range(1):
    model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

sgd = keras.optimizers.SGD(lr=0.005) #, clipnorm=1.)

model.compile(loss='mean_squared_error',
			  optimizer=sgd,
              metrics=['mse'])

#EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
callbacks_list = [earlystop]
callbacks_list=[] #remove this in order to have the early stop
start = time.time()
              

history = model.fit(input_train, output_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(input_validate, output_validate),
                    callbacks = callbacks_list)
end = time.time()


score = model.evaluate(input_test, output_test, verbose=0)  #use last 200 here?
print('Test MSE:', score[0])


#SVG(model_to_dot(model).create(prog='dot', format='svg'))




