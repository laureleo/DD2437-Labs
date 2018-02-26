# -*- coding: utf-8 -*-
# Same as test but with gaussian noise added
"""
Created on Mon Feb  5 09:26:48 2018

@author: s86852
"""
import random
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


# Noise 
mu = 0
sigma = 0.1

# Create noisy training and validation data

x = np.arange(0, 2*np.pi, 0.1)
f = np.sin(2*x)
f = f + np.random.normal(mu, sigma, f.shape)

c = list(zip(x, f))
random.shuffle(c)

train = c[:len(c)*2/3]
valid = c[len(c)*1/3:]

train_x, train_f = zip(*train)
train_x = np.array(train_x)
train_f = np.array(train_f)

valid_x, valid_f = zip(*valid)
valid_x = np.array(valid_x)
valid_f = np.array(valid_f)

test_x = x + 0.05
test_f = np.sin(2*test_x) 
test_f = test_f + np.random.normal(mu, sigma, test_f.shape)

#
#plt.figure("train plots")
#plt.plot(train_f)
#
#plt.figure("valid plots")
#plt.plot(valid_f)
#
#plt.figure("test plots")
#plt.plot(test_f)
#plt.show()
#
#
def train(epochs, eta, regStrength, hidden, hidden2):

# Setup parameters 
    batch_size = 100 #Smaller batches lead to more flucating gradients.
    regularization_strength = regStrength


# Define model layout (How many hidden layers, how many neurons in them, what type of activation function)
    model = Sequential()
    model.add(Dense(hidden, activation='tanh', W_regularizer=l1(regularization_strength) , input_dim=1)) 
    model.add(Dense(hidden2, activation='tanh', W_regularizer=l1(regularization_strength)))                                                                
    model.add(Dense(1, activation='linear'))                                                                


# Configure learning process
    model.compile( optimizer= keras.optimizers.SGD(lr = eta),  #Choose learning function
        loss = losses.mean_squared_error,           #Choose objective function
        metrics=['mean_squared_error']              #Choose metric to track
        )                         


# Setup early stopping and tensorboard
    earlystop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=5,
                                      verbose=1, mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = []


# Train on data and save info about the process into history
    history = model.fit(train_x, train_f,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(valid_x, valid_f),
        callbacks=callbacks_list)



# Evaluate model 
    score = model.evaluate(test_x, test_f, verbose=0)  #use last 200 here?

#Save graph for actual and ideal on test se
    predictions = model.predict(test_x)
        

#Print original time series as well as predicted time series
    plt.figure('Performance')
    plt.plot(predictions, label = 'first layer = {}, second layer = {}'.format(hidden, hidden2)) 
    legend = plt.legend(loc='upper right', shadow=True)
    plt.figure('Validation Error')
    plt.plot(history.history['val_mean_squared_error'], label = hidden)
    legend = plt.legend(loc='upper right', shadow=True)
    

# Return final validation loss
  #  return history.history['val_mean_squared_error'][-1]

# Return test score
    return score[0]

for i in range(1, 6):
    score = train(10000, 0.04, 0, i, 6-i)
    print(score)
plt.figure('Performance')
plt.plot(test_f, label = 'ideal', linewidth = 4)

plt.show()
