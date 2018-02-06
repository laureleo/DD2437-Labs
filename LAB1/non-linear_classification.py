import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import slp as slp
import mlp as mlp

		
# Number of data points
N = 100

# Class 1
mean_1 = [1, 1]
cov_1 = [[0.04, 0], [0, 0.8]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, N)
class_1 = np.c_[class_1, np.ones(N)]

# Class 2
mean_2 = [0, 0]
cov_2 = [[0.5, 0], [0, 0.1]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, N)
class_2 = np.c_[class_2, np.zeros(N)]


# Merge and shufffle data
data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(data)


colors = ['red', 'blue']
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2],
		            cmap=matplotlib.colors.ListedColormap(colors))
plt.axis('equal')
plt.show()

# Extract input matrix X and target vector T
data = np.transpose(data)
T = [data[2, :]]
X = np.delete(data, 2,0)

# Teach the mlp 
mlp = mlp.MLP(0.01, 10000, 1)
mlp.learn(X, T)

# Let it predict outputs from the input matrix
output = mlp.predict(X, T)[0]

#Calculate mean squared error
mse = ((X - T) ** 2).mean(axis=None)

#Calculate amount of missclassifcations
actual = np.around(mlp.predict(X, T)[0]) 
ideal = T[0]
error = actual - ideal
missclassifications = np.count_nonzero(error)

print("mean squared error = ", mse)
print("missclassifications are ", missclassifications)


#Stop condition
not_converged = 1

#counter until convergence
i = 0

while(not_converged):
	#Continue learning
	mlp.continue_learning(10)

	error = np.matrix.round(mlp.predict(X, X) - X)
	if (np.all(error == 0)):
				not_converged = 0
	i+= 10 
print(i)
