# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:47:40 2018

@author: Olivier
"""

from Hopfield import *
import numpy as np
from random import *
import matplotlib.pyplot as plt

N = 50 #number of patterns
T = 200 #number of units

plt.figure()
for rho_thresh in [0.1, 0.05, 0.01]:
    list = np.zeros((N, T))
    for i in range(N):
        x = []
        for u in range(T):
            if random()<rho_thresh:
                x.append(1)
            else:
                x.append(0)
        list[i]=x
        
    t_v = np.linspace(0,1, 20)
    corr_t = []
    for theta in t_v:
        can_learn = True
        n =1
        while can_learn and n < N:
            learn = list[:n, :]
            net = Hopfield(T)
            net.sparse_learning(learn)
            
            correct=0
            for x in learn:
                if np.array_equal(x, net.sparse_update(net.sparse_update(x, theta), theta)):
                    correct+=1
            if correct !=n:
                can_learn = False
                corr_t.append(n-1)
            else:
                n+=1
    
    label = "rho=" + str(rho_thresh)
    plt.plot(t_v, corr_t, label=label)
plt.legend()
plt.show()




