import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import slp as slp

		
# Number of data points
N = 5

# Class 1
mean_1 = [2, 2]
cov_1 = [[0.04, 0], [0, 0.08]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, N)
class_1 = np.c_[class_1, np.ones(N)]

# Class 2
mean_2 = [0, 0]
cov_2 = [[0.05, 0], [0, 0.1]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, N)
class_2 = np.c_[class_2, np.zeros(N)]


# Merge and shufffle data
data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(data)

# Plot

colors = ['red', 'blue']
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.axis('equal')
plt.show()

# Extract input matrix X and target vector T
T = data[:,2]
X = np.delete(data, 2, 1)

slp = slp.Perceptron(0.01, 100)

