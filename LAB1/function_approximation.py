import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlp import *
from sklearn.metrics import mean_squared_error
#from dd2437_LAB1 import perceptron

#Gaussian definition
def gaussian(x,y):
    return np.exp(-x*x*0.1) * np.exp(-y*y*0.1) - 0.5

#3D plot of the bell shaped Gaussian
x = np.arange(-5.0,5.0,0.5)
y = np.arange(-5.0,5.0,0.5)
X,Y = np.meshgrid(x, y) # grid of point
Z = gaussian(X, Y)

#Uncomment to plot the original function
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_wireframe(X,Y,Z)
#ax.set_title("Original Gaussian")

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

eta = 0.01
epochs = 1000
hidden = 5
for hidden in [ 3, 5,  10,  20, 25]:
    MSE=np.zeros(25)

    for n in range(1, 26):
        #idea but doesn't work cause double dim  shape
        resized_patterns = np.resize(perm_patterns,(2,n))
        resized_targets = np.resize(perm_targets,(1,n))
        
        trained_network = MLP(eta, epochs, hidden)
        trained_network.learn(resized_patterns, resized_targets)
    
        trained_network.forward_pass(patterns) # in some sort
        z_pred = trained_network.O_output
        z_true = perm_targets
        MSE[n-1]= mean_squared_error(z_true, z_pred)    
        
    plt.figure()
    title="Mean Squared errors for " + str(hidden) + " hidden layers"
    plt.plot(MSE)
    plt.title(title)
    plt.ylabel("n")
    plt.show()



