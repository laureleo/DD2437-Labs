# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:15:05 2018

@author: Olivier
"""
import numpy as np
from Hopfield import *
import matplotlib.pyplot as plt 

net = Hopfield(1024)
pict = np.loadtxt("pict.dat", dtype='i', delimiter=',')
pict = np.reshape(pict, (11, 1024))
learn = []
learn.append(pict[0]); learn.append(pict[1]), learn.append(pict[2])
net.hebbian_learning(learn, scale=False)

net.sequential_update(pict[9])

#Three patterns are stable
#for x in learn:
#    plt.figure()
#    plt.subplot(121)
#    xp = np.reshape(x, (32, 32))
#    plt.imshow(xp)
#    plt.subplot(122)
#    y = net.update_rule(x)
#    yp = np.reshape(y, (32, 32))
#    plt.imshow(yp)
#plt.show()

#p10 ?
plt.figure()
y = net.update_rule(pict[9])
plt.subplot(131)
plt.imshow(np.reshape(pict[9], (32, 32)))
plt.subplot(132)
plt.imshow(np.reshape(y, (32, 32)))
plt.subplot(133)
plt.imshow(np.reshape(pict[0], (32, 32)))
#
##p11
#plt.figure()
#plt.subplot(131)
#plt.imshow(np.reshape(pict[1], (32, 32)))
#plt.subplot(132)
#plt.imshow(np.reshape(pict[2], (32, 32)))
#plt.subplot(133)
#plt.imshow(np.reshape(pict[10], (32, 32)))
#
y = net.update_rule(pict[10])
inp = y
while not np.array_equal(inp,  net.update_rule(inp)):
    inp = net.update_rule(inp)
plt.figure()

plt.figure()
plt.subplot(221)
plt.imshow(np.reshape(pict[1], (32, 32)))
plt.subplot(222)
plt.imshow(np.reshape(pict[2], (32, 32)))
plt.subplot(223)
plt.imshow(np.reshape(pict[10], (32, 32)))
plt.subplot(224)
plt.imshow(np.resize(inp, (32, 32)))


