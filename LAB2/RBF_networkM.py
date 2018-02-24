# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:22:41 2018

@author: Olivier
"""

import numpy as np
import matplotlib.pyplot as plt

    

class RBF_network(object):
    
    def __init__(self, x, f, n, var=15, dim = 1):
        self.n = n
        self.dim = dim
        self.weights = np.random.randn(n)/100
        print('\nx:')
        print('mean', np.mean(x,0))
        print('max', np.max(x,0))
        print('min', np.min(x,0))
        diff_min_max = np.abs(np.max(x,0) - np.min(x,0))
        print('diff', diff_min_max)
        lower = np.min(x,0)-diff_min_max/4
        upper= np.max(x,0)+diff_min_max/4
        print('lower', lower)
        print('upper', upper)
        self.mean = np.zeros((self.n, dim))
        print(self.mean)
        if dim == 1:
            self.mean = np.linspace(lower, upper, self.n)
        else:
            for i in range(dim):
                self.mean[:,i] = np.linspace(lower[i], upper[i], self.n)
        print('mean', self.mean)
        self.variance = np.ones(n)*var #np.ones(n)/n for batch works well
        
    def phi(self, x, i):
        if self.dim == 1:
            return np.exp ( -(x-self.mean[i])**2/(2*self.variance[i]))
        else:
            phi_vec = np.zeros(self.dim)
            for dim in range(self.dim):
                phi_vec[dim] = np.exp ( -(x[dim]-self.mean[i,dim])**2/(2*self.variance[i]))
            print('phi_vec', phi_vec)
            return phi_vec
    
    def learning_batch(self, x, f):
        diff_min_max = np.abs(np.max(x) - np.min(x))
        self.mean = np.linspace(np.min(x)-diff_min_max/4, np.max(x)+diff_min_max/4, self.n)
        phi_mtx = np.array([[  self.phi(x[i], j)  for j in range(self.n)]    for i in range(len(x))])
        self.weights = np.array(np.linalg.lstsq(phi_mtx, f)[0])
        
    def learning_incr(self, x, f, eta, error_bound = 0.01, noite=10, CL = False, CL_iter = 3):
        #diff_min_max = np.abs(np.max(x) - np.min(x))*1
        #self.mean = np.linspace(np.min(x)-diff_min_max/4, np.max(x)+diff_min_max/4, self.n)

        #error = np.average(np.sum((np.sin(2*x) - self.output(x))**2))
        error = 1000 # Dummy 
        if CL:
            print('pre mean:', self.mean)
            count = 0
            eta_CL = 0.2
            print('\n -- Using CL with eta_CL = ', eta_CL,' -- ')
            rand_ints = np.random.randint(len(x), size=CL_iter)
            for i in rand_ints:
                if self.dim == 1:
                    data = x[i]
                    data_vec = np.ones(self.n)*data
                    dist = np.sqrt((data_vec-self.mean)**2)
                    d_min_i = np.argmin(dist) # Only one winner at the moment (some leaky learning should be implemented to avoid dead units)
                    self.mean[d_min_i] -= eta_CL*(self.mean[d_min_i]-data)
                else:
                    data = x[i,:]
                    dist = np.zeros(self.n)
                    for j in range(self.n):
                        dist[j] = np.linalg.norm(data - self.mean[j,:])
                    d_min_i = np.argmin(dist) # Only one winner at the moment (some leaky learning should be implemented to avoid dead units)
                    #print('eta_CL*(self.mean[d_min_i]-data)', eta_CL*(self.mean[d_min_i]-data))
                    self.mean[d_min_i] -= eta_CL*(self.mean[d_min_i]-data)
                #print(dist, d_min_i)
            print('post mean:', self.mean)
            
        ite=0
        while error > error_bound and ite < noite:
            for k in range(len(x)):
                if self.dim == 1:
                    phi_vect = np.array( [self.phi(x[k], i) for i in range(self.n)])
                    #error = 1/2 ( f[k] - np.sum(self.weights * phi_vect) )**2
                    d_w = eta* (f[k] - np.sum(self.weights * phi_vect)) * phi_vect 
                else:
                    print(self.dim)
                    phi_vect = np.array( [self.phi(x[k], i) for i in range(self.n)])
                    for dim in range(self.dim):
                        print('f[k,dim]', f[k,dim])
                        print('self.weights', self.weights)
                        print('phi_vect[:,dim]', phi_vect[:,dim])
                        d_w[dim] = eta* (f[k,dim] - np.sum(self.weights[:,dim] * phi_vect[:,dim])) * phi_vect[:,dim] 

                self.weights +=d_w
            #error = np.average(np.sum((f - self.output(x))**2))
            ite+=1
            
        
    def output(self, x):
        o = np.zeros(len(x))
        for k in range(len(x)):
            phi_vect = np.array( [self.phi(x[k], i) for i in range(self.n)])
            o[k]= np.sum(self.weights* phi_vect)
        return o
            
    
        
    
    
        
        
