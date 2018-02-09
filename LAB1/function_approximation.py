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
#don't permute like this
#perm_patterns = np.random.permutation(patterns)
#perm_targets = np.random.permutation(targets)


eta = 0.01
epochs = 1000
hidden_layers = [10];#[1, 3, 5, 10,  15, 20, 25]

MSE=np.zeros((len(hidden_layers), 25))
Ntimes = 1
for t in range(Ntimes):
    c=0
    indices = range(targets.shape[1])
    indices = np.random.permutation(indices)
    for hidden in hidden_layers:
        for n in [1, 2 , 5 , 10, 25, 75, 150, 250, patterns.shape[1]]:
            resized_patterns = np.zeros((2, n))
            resized_targets = np.zeros((1,n))
            for i in range(n):
                resized_patterns[0][i]=patterns[0][indices[i]]
                resized_patterns[1][i]=patterns[1][indices[i]]
                resized_targets[0][i]=targets[0][indices[i]]
            
            trained_network = MLP(eta, epochs, hidden)
            trained_network.learn(resized_patterns, resized_targets)
        
            trained_network.forward_pass(patterns) # in some sort
            z_pred = trained_network.O_output
            z_true = targets
            
            fig3 = plt.figure()
            ax = fig3.gca(projection='3d')
            ax.plot_wireframe(X,Y,z_pred.reshape((len(x), len(y))), color='r')
            ax.plot_wireframe(X,Y, z_true.reshape((len(x), len(y))))
            title= "approximation for n=" + str(n)
            ax.set_title(title)
            
            #mean squarred error of some sort
            #MSE[c][n-1]+=  mean_squared_error(z_true, z_pred) #/ np.linalg.norm(z_true - np.average(z_true))
            #normalized MSE ?
            #MSE[c][n-1]+= np.sum((z_true - z_pred)**2)/z_true.shape[1]
        c+=1
        
#    plt.figure()
#    title="Mean Squared errors for " + str(hidden) + " hidden layers"
#    plt.plot(MSE)
#    plt.title(title)
#    plt.xlabel("n")
#    plt.show()
MSE/=Ntimes
plt.figure()
plt.xlabel("n")
title="Mean Squared Errors"
for k in range(MSE.shape[0]):
    s = str(hidden_layers[k]) + " hidden layers"
    plt.plot(range(1, 26), MSE[k], label=s)
plt.legend()

plt.show()
    



