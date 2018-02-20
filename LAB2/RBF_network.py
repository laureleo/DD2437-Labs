# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:22:41 2018

@author: Olivier
"""

import numpy as np
import matplotlib.pyplot as plt

    

class RBF_network(object):
    
    def __init__(self,  n, div=15):
        self.weights = np.random.randn(n)/100
        self.mean = np.linspace(-1,1,n)
        self.variance = np.ones(n)/div #np.ones(n)/n for batch works well
        self.n = n
        
    def phi(self, x, i):
        return np.exp ( -(x-self.mean[i])**2/(2*self.variance[i]))
    
    def learning_batch(self, x, f):
        diff_min_max = np.abs(np.max(x) - np.min(x))
        self.mean = np.linspace(np.min(x)-diff_min_max/4, np.max(x)+diff_min_max/4, self.n)
        phi_mtx = np.array([[  self.phi(x[i], j)  for j in range(self.n)]    for i in range(len(x))])
        self.weights = np.array(np.linalg.lstsq(phi_mtx, f)[0])
        
    def learning_incr(self, x, f, eta ):
        diff_min_max = np.abs(np.max(x) - np.min(x))*4
        self.mean = np.linspace(np.min(x)-diff_min_max/4, np.max(x)+diff_min_max/4, self.n)
        for k in range(len(x)):
            phi_vect = np.array( [self.phi(x[k], i) for i in range(self.n)])
            #error = 1/2 ( f[k] - np.sum(self.weights * phi_vect) )**2
            
            d_w = eta* (f[k] - np.sum(self.weights * phi_vect)) * phi_vect 
            self.weights +=d_w
            
       
        
    def output(self, x):
        o = np.zeros(len(x))
        for k in range(len(x)):
            phi_vect = np.array( [self.phi(x[k], i) for i in range(self.n)])
            o[k]= np.sum(self.weights* phi_vect)
        return o
            
    
        
    
    
        
        