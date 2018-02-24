# -*- coding: utf-8 -*-
# Same as test but with gaussian noise added
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
import time


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

mu = 0
sigma = 0.18
noise = np.random.normal(mu, sigma, x.shape)
x = x + noise


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





def train(epochs, eta, regStrength, hidden, hidden2):

# Setup parameters 
    batch_size = 100 #Smaller batches lead to more flucating gradients.
    regularization_strength = regStrength


# Define model layout (How many hidden layers, how many neurons in them, what type of activation function)
    model = Sequential()
    model.add(Dense(hidden, activation='sigmoid', W_regularizer=l1(regularization_strength) , input_dim=5)) 
    #model.add(Dense(hidden2, activation='sigmoid', W_regularizer=l1(regularization_strength)))                                                                
    model.add(Dense(1, activation='linear'))                                                                


# Configure learning process
    model.compile( optimizer= keras.optimizers.SGD(lr = eta),  #Choose learning function
        loss = losses.mean_squared_error,           #Choose objective function
        metrics=['mean_squared_error']              #Choose metric to track
        )                         


# Setup early stopping and tensorboard
    earlystop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=5, 
                                      verbose=0, mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [earlystop]


# Train on data and save info about the process into history
    history = model.fit(input_train, output_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(input_validate, output_validate),
        callbacks=callbacks_list)



# Evaluate model 
    score = model.evaluate(input_test, output_test, verbose=0)  #use last 200 here?

#Save graph for actual and ideal on test se
    predictions = model.predict(input_test)
        

#Print original time series as well as predicted time series
    plt.figure('Performance')
    plt.plot(predictions, label = 'actual')
    plt.plot(output_test, label = 'ideal')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.figure('Validation Error')
    plt.plot(history.history['val_mean_squared_error'], label = hidden)
    legend = plt.legend(loc='upper right', shadow=True)
    

# Return final validation loss
  #  return history.history['val_mean_squared_error'][-1]

# Return test score
    return score[0]

def get_average_validation_errors():
    iterations = 1
    hiddensize = 0

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 1, hiddensize)
        print(i)
    print('\n 1 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 2, hiddensize)
        print(i)
    print('\n 2 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 3, hiddensize)
        print(i)
    print('\n 3 avg val_loss is')

    print(s/iterations)
    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, hiddensize)
        print(i)
    print('\n 4 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 5, hiddensize)
        print(i)
    print('\n 5 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 6, hiddensize)
        print(i)
    print('\n 6 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 7, hiddensize)
        print(i)
    print('\n 7 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 8, hiddensize)
        print(i)
    print('\n 8 avg val_loss is')
    print(s/iterations)


#Only applies to the first layer
#Don't forget to turn of the second layer
#Don't forget to only return validation error
def get_effect_of_regularization(strength):
    hiddensize = 0
    print('Strenght is ', strength)
    print('\n 1')
    print(train(10000, 0.02, strength, 1, hiddensize))
    print('\n 2')
    print(train(10000, 0.02, strength, 2, hiddensize))
    print('\n 3')
    print(train(10000, 0.02, strength, 3, hiddensize))
    print('\n 4')
    print(train(10000, 0.02, strength, 4, hiddensize))
    print('\n 5')
    print(train(10000, 0.02, strength, 5, hiddensize))
    print('\n 6')
    print(train(10000, 0.02, strength, 6, hiddensize))
    print('\n 7')
    print(train(10000, 0.02, strength, 7, hiddensize))
    print('\n 8')
    print(train(10000, 0.02, strength, 8, hiddensize))



#Don't forget to turn on the second layer
#Don't forget to only return validation error
def get_effect_of_regularization(strength, hiddensize):
    print('Strenght is ', strength)
    print('\n 1')
    print(train(10000, 0.02, strength, 1, hiddensize))
    print('\n 2')
    print(train(10000, 0.02, strength, 2, hiddensize))
    print('\n 3')
    print(train(10000, 0.02, strength, 3, hiddensize))
    print('\n 4')
    print(train(10000, 0.02, strength, 4, hiddensize))
    print('\n 5')
    print(train(10000, 0.02, strength, 5, hiddensize))
    print('\n 6')
    print(train(10000, 0.02, strength, 6, hiddensize))
    print('\n 7')
    print(train(10000, 0.02, strength, 7, hiddensize))
    print('\n 8')
    print(train(10000, 0.02, strength, 8, hiddensize))


def get_all_regEffects():
    get_effect_of_regularization(0.001, 0)
    get_effect_of_regularization(0.01, 0)
    get_effect_of_regularization(0.1, 0)
    get_effect_of_regularization(1, 0)


#Don't forget to turn on the second layer in the model
#Don't forget to set train return value to be validation error
def get_average_validation_errors_second_layer():
    iterations = 1

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 1)
        print(i)
    print('\n 1 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 2)
        print(i)
    print('\n 2 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 3)
        print(i)
    print('\n 3 avg val_loss is')

    print(s/iterations)
    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 4)
        print(i)
    print('\n 4 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 5)
        print(i)
    print('\n 5 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 6)
        print(i)
    print('\n 6 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 7)
        print(i)
    print('\n 7 avg val_loss is')
    print(s/iterations)

    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 8)
        print(i)
    print('\n 8 avg val_loss is')
    print(s/iterations)

#Don't forget to turn off the second layer in the model
#Don't forget to turn on noise
#Don't forget to set train return value to be test error
def test_best_1_layer_with_noise():
    iterations = 10
    s = 0
    for i in range(iterations):
        s = s + train(10000, 0.02, 0, 4, 2)
        print(i)
    print('\n avg val_loss is')
    print(s/iterations)

def checktime(hidden):
    start = time.time()
    train(1000, 0.02, 0, hidden, hidden)
    end = time.time()
    print('\n')
    print(end - start)

for i in range(1, 8):
    checktime(i)
        
