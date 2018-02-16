# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:04:25 2018

@author: mathias
"""

import numpy as np
	

class Perceptron(object):
     
     def __init__(self, eta, epochs, type_p = 'delta'):
          self.eta = eta
          self.epochs = epochs
          self.type_p = type_p
          self.C = np.zeros(self.epochs)  # Missclassifications
          
     def run_pct(self, X, T):
          self.setup(X, T)
          self.init_weights()
          if self.type_p == 'delta':
               self.delta_lrn()
          elif self.type_p == 'perc':
               self.perc_lrn()
          
     def setup(self, X, T):
          #Convert input into a format numpy can deal with
          self.T = np.array(T)
          self.X = np.array(X)
          
          #Fill a one-dimensional array (with the same number of cols as X) with 1:s. This is the bias vector
          bias = np.ones(self.X.shape[1])
          
          #Append the bias vector as the last row in the input matrix
          self.X = np.vstack((self.X, bias))

          
     def init_weights(self):
          W_rows = output_dimensionality = self.T.shape[0]
          W_cols = input_dimensionality = self.X.shape[0]
          
          #Initialize a weight matrix with as many rows as the output dimensionality, as many cols as input dimensionality
          self.W = np.zeros(shape=(W_rows, W_cols))
          
          #Fill each row with values drawn randomly from a normal distribution with a stdev of 0.1
          for i in range (W_rows):
               self.W[i] = np.random.normal(0, 0.1, W_cols)
               
     def predict(self, X, T):
          # Use the weights learned to check for activation
          self.setup(X, T)
          activations = self.pcnfwd()
          return(activations)
          
     def pcnfwd(self):
          activations = np.dot(self.W, self.X)
          return np.where(activations > 0, 1, 0)
          
     def delta_lrn(self):
          for i in range(self.epochs):
               activations = self.pcnfwd()
               if np.array_equal(activations, self.T>=0):
                    #break    # Go till all are correctly classified
                    pass      # Go for all epochs
               
               # Net input
               net_input = np.dot(self.W, self.X)
               error = net_input - self.T
               delta_W = -1 * self.eta * np.dot(error, np.transpose(self.X))
               self.W += delta_W
               self.C[i] = np.sum((activations==1) != (self.T==1)) # Numer of missclassified this epoch
               
     def perc_lrn(self):
          # https://en.wikipedia.org/wiki/Perceptron
          for i in range(self.epochs):
               missclass_epoch = 0
               for j in range(np.shape(self.X)[1]):
                    # Net input
                    X = self.X[:, j]
                    T = self.T[0, j]
                    
                    net_input = np.dot(self.W, X)
                    
                    # activation
                    if net_input >= 0:
                         activation = 1
                    else:
                         activation = 0
                    
                    target_acti = T>=0
                    if activation == T>=0:
                         pass
                    elif activation != target_acti:
                         missclass_epoch += 1
                         error = T - net_input
                         for k in range(np.shape(self.W)[1]):
                              self.W[0,k] += self.eta * error[0] * self.X[:, j][k]             
               self.C[i] = missclass_epoch # Numer of missclassified this epoch
               


#Example use
"""
in1 =[[1,1,-1,-1],
      [1,-1,1,-1]]

out1 = [[1,1,1,-1]]

eta = 0.001
epochs = 100
pct = Perceptron(eta, epochs,'perc')
pct.run_pct(in1, out1)
print(pct.predict(in1, out1))
plt.plot(pct.C)
"""