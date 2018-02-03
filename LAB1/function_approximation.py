import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlp import *

#Gaussian definition
def gaussian(x,y):
    return np.exp(-x*x*0.1) * np.exp(-y*y*0.1) - 0.5

#3D plot of the bell shaped Gaussian
x = np.arange(-5.0,5.0,0.5)
y = np.arange(-5.0,5.0,0.5)
X,Y = np.meshgrid(x, y) # grid of point
Z = gaussian(X, Y)
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(X,Y,Z)
ax.plot_wireframe(X,Y,Z)
ax.set_title("Original Gaussian")

#patterns (in) and targets(out) matrices
ndata = len(x)*len(y)
targets = Z.reshape(1,ndata).copy() 
xx, yy = np.meshgrid(x, y) # grid of point
patterns = np.concatenate((xx.reshape( 1, ndata).copy(), yy.reshape( 1, ndata).copy())) #2xndata


###Part 3.3.2
#eta = 0.01
#epochs = 1000
#hidden = 10
#MLP = MLP(eta, epochs, hidden, draw='True', x_draw=x, y_draw=y)
#MLP.learn(patterns, targets)
#prediction = MLP.predict(patterns, targets)
#
#fig3 = plt.figure()
#ax = fig3.gca(projection='3d')
#zz = prediction.reshape((len(x), len(y))) 
#ax.plot_wireframe(X,Y,zz)
#ax.set_title("Final approximation")

###Part 3.3.3
#Permutations of patterns and target vectors
perm_patterns = np.random.permutation(patterns)
perm_targets = np.random.permutation(targets)

for n in range(1, 26):
    patterns = perm_patterns[:n]
    targets = perm_targets[:n]
    
    training_network = MLP(eta, epochs, hidden)
    training_network.learn(patterns, targets)
    
    all_network = MLP(eta, epochs, hidden)
    all_network.learn(perm_patterns, perm_targets)
    
    
    
