# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:23:40 2018

@author: Olivier
"""
import matplotlib.pyplot as plt
import numpy as np
from RBF_network import RBF_network
net = RBF_network(100, 0.01)


x = np.arange(0, 2*np.pi, 0.1)
f = np.sin(2*x)

def func(x):
    if x >=0:
        return 1
    else:
        return -1
#f = np.vectorize(func)(f)
#net.learning_batch(x,f)
net.learning_incr(x, f, 0.004, 0.001, 1000) #O.O41 is a good one for 100
#0.009 for 1000 nodes
#0.0009 for 10000 nodes
#rangeE = np.linspace(0.0001, 0.001, 10)
##rangeV = np.linspace(1, 100, 50)
#MSE = np.zeros(10)
#i=0
#for eta in rangeE:
#    np.random.seed(9001)
#    net = RBF_network(10000)
#    net.learning_incr(x, f, eta)
#    test_x = x+  0.05
#    o = net.output(test_x)
#    MSE[i] = np.average(np.sum((np.sin(2*test_x) - o)**2))
#    i+=1
#plt.figure()
#plt.plot(rangeE, MSE)
#plt.show()

plt.figure()
test_x = x + 0.05
o = net.output(test_x)
plt.plot(test_x, np.sin(2*test_x), linewidth=5)
plt.plot(test_x, o)
absolute_residual_error = np.average(np.abs(np.sin(2*test_x) - o))
plt.show()