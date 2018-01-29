import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
ax.plot_surface(X,Y,Z)

#Matrices pattern and targets
ndata = len(x)*len(y)
targets = Z.reshape(1,ndata).copy()
xx, yy = np.meshgrid(x, y) # grid of point
patterns = [xx.reshape( 1, ndata), yy.reshape( 1, ndata)]