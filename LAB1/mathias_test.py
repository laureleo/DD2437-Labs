# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:45:20 2018

@author: mathias
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from drawnow import drawnow

# Part 1 general
N = 100 # Num obs per class
mu1 = 1
mu2 = 8
var = 2

# Class 1
mean_1 = [mu1, mu1]
cov_1 = [[var, 0], [0, var]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, N)
class_1 = np.c_[class_1, np.ones(N)]  #adding bias and class

# Class 2
mean_2 = [mu2, mu2]
cov_2 = [[var, 0], [0, var]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, N)
class_2 = np.c_[class_2, -np.ones(N)]  #adding bias and class

# Merge and shufffle data
data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(data)
data = np.transpose(data)

##
from slp_m import Perceptron

seed = np.random.randint(1,10000)

for dummy in ['perc','delta']:
     ## For testing both types of learning on the same seed.
     print('----------------------------------------------')
     print('Method:', dummy, ', with seed:', seed)
     np.random.seed(seed)
     perceptron = Perceptron(0.0001, 1000, dummy)
     perceptron.run_pct(data[:-1, :], data[2:, :]) # train
     activations = perceptron.predict(data[:-1, :], data[2:, :])
     print('Correctly classified [', np.sum((data[2:, :]>=0) == activations), '/', np.size(data[2:, :]), '] points')
     print('----------------------------------------------')
     
     
     # Plot data
     colors = ['red', 'blue']
     plt.scatter(data[0, :], data[1, :], c=data[2, :],
                 cmap=matplotlib.colors.ListedColormap(colors))
     x = np.linspace(mu1 - 1, mu2 + 1, num=5)
     plt.plot(x, -(perceptron.W[0][0]/perceptron.W[0][1])*x - (perceptron.W[0][2]/perceptron.W[0][1]))
     plt.axis('equal')
     plt.show()

     plt.plot(perceptron.C)
     plt.show()

print('----------------------------------------------')

