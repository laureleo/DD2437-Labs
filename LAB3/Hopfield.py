# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:08:39 2018

@author: Olivier
"""
import numpy as np 

class Hopfield(object):
    def __init__(self, N): 
        self.N = N #number of units
        self.W = np.zeros((N,N))
        
    def Hebbian_learning(self, patterns_list, scale=True):
        P = len(patterns_list)
        for i in range(self.N):
            for j in range(self.N):
                for mu in range(P):
                    self.W[i][j]+= patterns_list[mu][i]*patterns_list[mu][j]
                if scale:
                    self.W[i][j]/=self.N
            
    #def Network_recall(self, list_x):
    def sign(self, x):
        if x >=0:
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
    
        
    
    
##===========================================================================##

l = []
net = Hopfield(8)
x1=[-1, -1, 1, -1, 1, -1, -1, 1]
x2=[-1, -1, -1, -1, -1, 1, -1, -1]
x3=[-1, 1, 1, -1, -1, 1, -1, 1]
l.append(x1); l.append( x2); l.append( x3)

net.Hebbian_learning(l, scale=False)
for x in l:
    print("vector {} and update rule {}".format(x, net.update_rule(x)))
    print("diff={}".format(x - net.update_rule(x)))