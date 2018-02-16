# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:26:48 2018

@author: s86852
"""
import tensorflow as ts
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import losses
import time
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.regularizers import l2, l1


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

# Select data for training, validating and testing
t_start = 300; t_end = 1500

input_ts = np.array([x[range(t_start-20, t_end-20)],
                      x[range(t_start-15, t_end-15)],
                      x[range(t_start-10, t_end-10)],
                      x[range(t_start-5, t_end-5)],
                      x[range(t_start, t_end)]])

output_ts = np.array([x[range(t_start+5, t_end+5)]])

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





# Loop for checking multiple runs with differing network structures
for i in range (1):

# Setup parameters 
    batch_size = 100 #Smaller batches lead to more flucating gradients.
    epochs = 1000
    eta = 0.01
    regularization_strength = 0.001



# Define model layout (How many hidden layers, how many neurons in them, what type of activation function)
    model = Sequential()
    model.add(Dense(8, activation='sigmoid', W_regularizer=l2(regularization_strength) , input_dim=5)) 
    #model.add(Dense(8, activation='sigmoid', W_regularizer=l2(regularization_strength)))                                                                
    model.add(Dense(1, activation='linear'))                                                                


# Configure learning process
    model.compile(
        optimizer= keras.optimizers.SGD(lr = eta),  #Choose learning function
        loss = losses.mean_squared_error,           #Choose objective function
        metrics=['mean_squared_error']              #Choose metric to track
        )                         


# Setup early stopping and tensorboard
    earlystop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=5, 
                                      verbose=1, mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [earlystop, tensorboard]


# Train on data and save info about the process into history
    history = model.fit(input_train, output_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(input_validate, output_validate),
        callbacks=callbacks_list)



# Evaluate model 
    score = model.evaluate(input_test, output_test, verbose=1)  #use last 200 here?
    print('Test MSE:', score[0])

#Save graph for actual and ideal on test se
predictions = model.predict(input_test)
    

#Print original time series as well as predicted time series
plt.figure('MLP Prediction')
plt.plot(predictions)
plt.figure('True Values')
plt.plot(output_test)

plt.show()
#To view the training process
#Delete the logs folder in this folder
#Run this file
#In the terminal/shell, run tensorboard --logdir ./logs
#Visit whe website indicated



