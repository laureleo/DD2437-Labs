# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:16:22 2018
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import solve


# Part 1 general
N = 10 # Num obs per class

# Class 1
mean_1 = [2, 2]
cov_1 = [[0.04, 0], [0, 0.08]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, N)
class_1 = np.c_[class_1, np.ones(N)]

# Class 2
mean_2 = [0, 0]
cov_2 = [[0.05, 0], [0, 0.1]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, N)
class_2 = np.c_[class_2, np.zeros(N)]

# Merge and shufffle data
data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(data)

# Plot

colors = ['red', 'blue']
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.axis('equal')
plt.show()

# Perceptron Learning

class perceptron:
    def __init__(self, inputs, targets):
        if np.ndim(inputs)>1:
            self.nIn = np.shape(inputs)[1]
        else:
            nIn = 1
        
        if np.ndim(targets)>1:
            self.nOut = np.shape(targets)[1]
        else:
             self.nOut = 1
             
        self.nData = np.shape(inputs)[0]
	
		 # Initialise network
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self, inputs, targets, eta, nIterations, verbose):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, np.ones((self.nData, 1))), axis=1)
        # Training
        if verbose:
            print('-----------------------------')
            print('Inputs: \n', inputs)
            print('-----------------------------')
            print('Targets: \n', targets)
        for n in range(nIterations):
            self.activations = self.pcnfwd(inputs)
            self.weights -= eta*np.dot(np.transpose(inputs), self.activations-targets)
            if np.array_equal(targets, self.activations):
                break
        if verbose:    
            print('-----------------------------')
            if n+1 < nIterations:
                print('Done in ', n+1, ' Iterations')
                print('Activations \n', self.activations)
            else:
                print('Max itersations of ', n+1, ' reached')
            print('-----------------------------')
        else:
            print('Done in ', n+1, ' Iterations')
            
    def deltatrain(self, inputs, targets, eta, nIterations, verbose):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = np.concatenate((np.transpose(inputs), np.ones((1,self.nData))), axis=0)
        # Training
        if verbose:
            print('-----------------------------')
            print('Inputs: \n', inputs)
            print('-----------------------------')
            print('Targets: \n', targets)
        for n in range(nIterations):
            self.activations = self.pcnfwd(np.transpose(inputs))
            print('-----------debug-------------')
            #print(self.weights)
            #print(inputs)
            #print(np.dot(np.transpose(self.weights),inputs))
            #print('-----------------------------')
            #print((np.dot(np.transpose(self.weights), inputs) - np.transpose(targets)))
            #print(np.transpose(inputs))
            #print()
            self.weights -= np.transpose(eta*np.dot((np.dot(np.transpose(self.weights), inputs) - np.transpose(targets)),
                                       np.transpose(inputs)))
            #print(self.pcnfwd(np.transpose(inputs)))
            print(targets)
            print(self.activations)
            if np.array_equal(targets, self.activations):
                break
        if verbose:    
            print('-----------------------------')
            if n+1 < nIterations:
                print('Done in ', n+1, ' Iterations')
                print('Activations \n', self.activations)
            else:
                print('Max itersations of ', n+1, ' reached')
            print('-----------------------------')
        else:
            print('Done in ', n+1, ' Iterations')        

    def pcnfwd(self, inputs):
        """ Run the network forward """
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)
    
    def test_data(self, inputs):
        inputs = np.append(inputs, [1])
        print(inputs)
        activation_test = self.pcnfwd(inputs)
        print(activation_test)
        

pct_obj = perceptron(data[:, :-1], data[:, 2:])
pct_obj.pcntrain(data[:, :-1], data[:, 2:], 0.01, 1000, 1)
print('Weights: \n', pct_obj.weights)

pct_obj.deltatrain(data[:, :-1], data[:, 2:], 0.001, 1000, 1)
print('Weights: \n', pct_obj.weights)


pct_obj.test_data([0, 0])
pct_obj.test_data([2, 2])

        
        



