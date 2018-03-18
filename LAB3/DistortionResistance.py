# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:59:47 2018

@author: Olivier
"""

import numpy as np
from Hopfield import *
from random import * 

net = Hopfield(1024)
pict = np.loadtxt("pict.dat", dtype='i', delimiter=',')
pict = np.reshape(pict, (11, 1024))
learn = []
learn.append(pict[0]); learn.append(pict[1]), learn.append(pict[2])
net.hebbian_learning(learn, scale=False)

#up to 40% noise still converges to p1 and then converges to another fix point
noise = 0
while noise < 1024:
    noise+=102
    if noise > 1024:
        noise = 1024
        
    p1 = np.copy(learn[0])
    for idx in range(noise):
        p1[idx]= -p1[idx]
    
    nb_it=1
    inp = np.copy(p1)
    while (not np.array_equal(inp, net.update_rule(inp)) ) and nb_it < 10:
        nb_it+=1
        inp = net.update_rule(inp)
    
    if noise%102 ==0:
        print("for {} added noise".format(noise/1024*100))
        print("p1 succesfully retrieved") if np.array_equal(inp, learn[0]) else print("p1 not successfully retrieved")
        print("in {} iterations".format(nb_it))
    
    