# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:23:40 2018

@author: Olivier
"""
import matplotlib.pyplot as plt
import numpy as np
from RBF_network import RBF_network
net = RBF_network(100)


x = np.arange(0, 2*np.pi, 0.01)
f = np.sin(2*x)

net.learning_incr(x, f, 0.007)
#rangeE = [0.007]#np.linspace(0.001, 0.01, 100)
#rangeV = np.linspace(1, 100, 50)
#MSE = np.zeros(50)
#i=0
#for div in rangeV:
#    np.random.seed(9001)
#    net = RBF_network(100, div)
#    net.learning_incr(x, f, 0.007)
#    test_x = x+  0.05
#    o = net.output(test_x)
#    MSE[i] = np.average(np.sum((np.sin(2*test_x) - o)**2))
#    i+=1
#
#plt.plot(rangeV, MSE)
#plt.show()

plt.figure()
test_x = x
o = net.output(test_x)
plt.plot(test_x, np.sin(2*test_x), linewidth=5)
plt.plot(test_x, o)
plt.show()