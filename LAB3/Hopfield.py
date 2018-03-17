# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:08:39 2018

@author: Olivier
"""
import numpy as np 
from math import *
import matplotlib.pyplot as plt
from random import *

class Hopfield(object):
    def __init__(self, N): 
        self.N = N #number of units
        self.W = np.zeros((N,N))
        
    #Learning made with a list of patterns and with a possibility not to scale the weights
    def hebbian_learning(self, patterns_list, scale=False): #false if just sign
        P = len(patterns_list)
        for i in range(self.N):
            for j in range(self.N):
                for mu in range(P):
                    self.W[i][j]+= patterns_list[mu][i]*patterns_list[mu][j]
                if scale:
                    self.W[i][j]/=self.N
        
            
    #def Network_recall(self, list_x):
    def sign(self, x):
        if x >= 0:
            return(1)
        else: 
            return(-1)
    
    def update_rule(self, x):
        x2 = np.copy(x)
        for i in range(len(x)):
            x2[i]=0
            for j in range(len(x)):
                x2[i]+=self.W[i][j]*x[j]
            x2[i]=self.sign(x2[i])
        return(x2)
    
    def sequential_update(self, x):
        it = 0
        out = np.copy(x) 
        while it < 5501 :
            idx = randint(0, self.N-1)
            s =0
            for j in range(len(x)):
                s+=self.W[idx][j]*out[j]
            out[idx]=self.sign(s)
            
            if it%500==0:
                print("energy during iteration is:{}".format(self.energy(out)))
#uncomment to plot every 500 it
#            if it%500 == 0:
#                plt.figure()
#                plt.imshow(np.reshape(out, (32, 32)))
            it+=1
        
    
    def energy(self, x):
        E = 0
        for i in range(self.N):
            for j in range(self.N):
                E+=self.W[i][j]*x[i]*x[j]
        return( -E)
    
    
##===========================================================================##


###2.2 
#l = []
#net = Hopfield(8)
#x1=[-1, -1, 1, -1, 1, -1, -1, 1]
#x2=[-1, -1, -1, -1, -1, 1, -1, -1]
#x3=[-1, 1, 1, -1, -1, 1, -1, 1]
#l.append(x1); l.append( x2); l.append( x3)
#
#net.hebbian_learning(l, scale=False)
##for x in l:
##    print("vector {} and update rule {}".format(x, net.update_rule(x)))
##    print("diff={}".format(x - net.update_rule(x)))
#    
###3.1 Convergence and attractors
#x1d=[ 1, -1, 1, -1, 1, -1, -1, 1] #converges to x1
#x2d=[ 1, 1, -1, -1, -1, 1, -1, -1] #converges to smth unknown
#x3d=[ 1, 1, 1, -1, 1, 1, -1, 1] #converges to x3
#
#l2 = [x1d, x2d, x3d]
#for x in l2:
#    entry = x
#    out = net.update_rule(entry)
#    while not np.array_equal(entry, out):
#        entry =out
#        out = net.update_rule(entry)
#    print("vec:{} and fix point:{}".format(x, out))
# 
##How many attractors are there in this network
#attractors = [x1, x2, x3]
##very greedy method to find the attractors by trying everypossibility
#trial = -np.ones(8)
#trials = []
#a = 1; b =1; c=1; d=1; e=1; g=1; f=1; h=1
#for i in range(2):
#    a =-a
#    for j in range(2):
#        b=-b
#        for k in range(2):
#            c=-c
#            for l in range(2):
#                d=-d
#                for m in range(2):
#                    e=-e
#                    for n in range(2):
#                        f=-f
#                        for o in range(2):
#                            g=-g
#                            for p in range(2):
#                                h=-h
#                                trials.append([a,b,c,d,e,f,g,h])
#
#
#
#n =0
#for trial in trials:
#    if (np.array_equal(net.update_rule(trial), trial)):
#        if trial not in attractors:
#            attractors.append(trial)
#
#
#print("the attractors are")
#for x in attractors:
#    print(x)
#    
#
#
