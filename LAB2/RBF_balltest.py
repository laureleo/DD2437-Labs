# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:23:40 2018

@author: Olivier
"""
import matplotlib.pyplot as plt
import numpy as np
from RBF_network import RBF_network
#net = RBF_network(20, 0.01)

data_train = np.loadtxt('ballist.dat')
#print('\n data_train: \n', data_train)

x = data_train[:, 0:2]
#print('\n x: \n', x)

f = data_train[:, 2:]
#print('\n f: \n',f)

"""
x = np.arange(0, 2*np.pi, 0.1)
f = np.sin(2*x)
"""

nunits = 20

net = RBF_network(x, f, nunits, (2*np.pi/(1*nunits))**2, dim = 2)
net.learning_incr(x, f, 0.04, 0.001, 100, True, CL_iter = 100)

# Testing
#print(net.weights)



"""
plt.figure()
test_x = x + 0.05
o = net.output(test_x)
plt.plot(test_x, np.sin(2*test_x), linewidth=5)
plt.plot(test_x, o)
absolute_residual_error = np.average(np.abs(np.sin(2*test_x) - o))
print('error', absolute_residual_error)
plt.show()
"""
