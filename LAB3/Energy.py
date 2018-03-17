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