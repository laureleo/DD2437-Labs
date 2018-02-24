# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:23:40 2018

@author: Olivier
"""
import matplotlib.pyplot as plt
import numpy as np
from RBF_network import RBF_network
#net = RBF_network(20, 0.01)

x = np.arange(0, 2*np.pi, 0.1)
f = np.sin(2*x)

nunits = 20

net = RBF_network(x,f, nunits, (2*np.pi/(1*nunits))**2, dim = 1)
net.learning_incr(x, f, 0.04, 0.001, 100, True, CL_iter = 100)

# Testing
#


plt.figure()
test_x = x + 0.05
o = net.output(test_x)
plt.plot(test_x, np.sin(2*test_x), linewidth=5)
plt.plot(test_x, o)
absolute_residual_error = np.average(np.abs(np.sin(2*test_x) - o))
print('error', absolute_residual_error)
plt.show()

