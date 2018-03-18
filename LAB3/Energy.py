# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:35:12 2018

@author: Olivier
"""

import numpy as np
from Hopfield import *

net = Hopfield(1024)
pict = np.loadtxt("pict.dat", dtype='i', delimiter=',')
pict = np.reshape(pict, (11, 1024))
learn = []
learn.append(pict[0]); learn.append(pict[1]), learn.append(pict[2])
net.hebbian_learning(learn, scale=False)

for x in learn:
    print ("Energy of attractor is:{}".format(net.energy(x)))
    
print("Energy of p10 is:{} and energy of p11 is:{}".format(net.energy(pict[9]), net.energy(pict[10])))

net.sequential_update(pict[9])


#normally distributed
#doesn't always converge depending on init
net2 = Hopfield(8)
net2.random_init()
x1=[ 1, -1, 1, -1, 1, -1, -1, 1]
nb = 0
inp = np.copy(x1)
bound_it=100
while (not np.array_equal(inp, net2.update_rule(inp))) and nb < bound_it:
    inp = net2.update_rule(inp)
    nb+=1
print(nb)

#WHen obtaining a symmetric matrix off of it, almost instantly converges
#Cause same weight from i to j and to j to i so less likely to get stuck in some place 
if nb==bound_it:
    nb=0
    net2.W = 1/2*(net2.W * np.transpose(net2.W))
while (not np.array_equal(inp, net2.update_rule(inp))) and nb < bound_it:
    inp = net2.update_rule(inp)
    nb+=1
print(nb)