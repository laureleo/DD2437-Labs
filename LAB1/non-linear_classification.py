import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import slp as slp
import mlp as mlp

		
# CREATE TRAINING DATA

# Number of data points
N = 100

# Class 1
mean_1 = [2, 2]
cov_1 = [[0.5, 0], [0, 1]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, N)
class_1 = np.c_[class_1, np.ones(N)]

# Class 2
mean_2 = [0, 0]
cov_2 = [[0.5, 0], [0, 1]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, N)
class_2 = np.c_[class_2, np.zeros(N)]


# Merge and shufffle training data
data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(data)





# CREATE TEST DATA

# Number of data points
M = N

# Class 1
mean_1 = [2, 2]
cov_1 = [[0.5, 0], [0, 1]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, M)
class_1 = np.c_[class_1, np.ones(M)]

# Class 2
mean_2 = [0, 0]
cov_2 = [[0.5, 0], [0, 1]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, M)
class_2 = np.c_[class_2, np.zeros(M)]


# Merge and shufffle training data
test_data = np.concatenate((class_1, class_2), axis=0)
np.random.shuffle(test_data)





# PLOT DATA DISTRIBUTION

total = np.concatenate((data, test_data), axis = 0)

# Draw sample space
plt.figure(1)
colors = ['red', 'blue']
plt.scatter(total[:, 0], total[:, 1], c=total[:, 2],
		            cmap=matplotlib.colors.ListedColormap(colors))
	




#SEPEARATE INPUT AND TARGET OF TRAINING DATA

data = np.transpose(data)
T = [data[2, :]]
X = np.delete(data, 2,0)


#SEPEARATE INPUT AND TARGET OF TEST DATA

test_data = np.transpose(test_data)
T2 = [test_data[2, :]]
X2 = np.delete(test_data, 2,0)





# TRAIN THE NETWORK, CHECK PREDICTIONS ON UNSEEN DATA AND ADD THEM TO PLOT

# Initialize the mlp
network= mlp.MLP(0.01, 1, 1)
network.learn(X, T)

#Setup 
i = 0
not_converged = 1
mse_list = []
mis_list = []
mse_list2 = []
mis_list2 = []


while(not_converged):
	network.continue_learning(1)

	#Predict mse on unseen test data
	output2 = network.predict(X2, T2)[0]
	mse2 = ((output2 - T2) ** 2).mean(axis=None)
	mse_list2.append(mse2)

	#Predict classification error on unseen test data
	actual2 = np.around(network.predict(X2, T2)[0]) 
	ideal2 = T2[0]
	error2 = actual2 - ideal2
	missclassifications2 = np.count_nonzero(error2)
	mis_list2.append(missclassifications2)

	#Calculate and append mean squared error
	output = network.predict(X, T)[0]
	mse = ((output - T) ** 2).mean(axis=None)
	mse_list.append(mse)

	#Count and append missclassifications
	actual = np.around(network.predict(X, T)[0]) 
	ideal = T[0]
	error = actual - ideal
	missclassifications = np.count_nonzero(error)
	mis_list.append(missclassifications)

	#Convergence check
	if (mse < 0.1):
		not_converged = 0

	i += 1

#Plot values
plt.figure(2)
plt.subplot(221)
plt.ylabel('TRAIN Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list)

plt.subplot(222)
plt.ylabel('TRAIN Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list, label = '1 hidden')

plt.subplot(223)
plt.ylabel('TEST Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list2)

plt.subplot(224)
plt.ylabel('TEST Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list2, label = '1 hidden')


# TRAIN THE NETWORK, CHECK PREDICTIONS ON UNSEEN DATA AND ADD THEM TO PLOT

# Initialize the mlp
network= mlp.MLP(0.01, 1, 2)
network.learn(X, T)

#Setup 
i = 0
not_converged = 1
mse_list = []
mis_list = []
mse_list2 = []
mis_list2 = []


while(not_converged):
	network.continue_learning(1)

	#Predict mse on unseen test data
	output2 = network.predict(X2, T2)[0]
	mse2 = ((output2 - T2) ** 2).mean(axis=None)
	mse_list2.append(mse2)

	#Predict classification error on unseen test data
	actual2 = np.around(network.predict(X2, T2)[0]) 
	ideal2 = T2[0]
	error2 = actual2 - ideal2
	missclassifications2 = np.count_nonzero(error2)
	mis_list2.append(missclassifications2)

	#Calculate and append mean squared error
	output = network.predict(X, T)[0]
	mse = ((output - T) ** 2).mean(axis=None)
	mse_list.append(mse)

	#Count and append missclassifications
	actual = np.around(network.predict(X, T)[0]) 
	ideal = T[0]
	error = actual - ideal
	missclassifications = np.count_nonzero(error)
	mis_list.append(missclassifications)

	#Convergence check
	if (mse < 0.1):
		not_converged = 0

	i += 1

#Plot values
plt.figure(2)
plt.subplot(221)
plt.ylabel('TRAIN Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list)

plt.subplot(222)
plt.ylabel('TRAIN Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list, label = '2 hidden')

plt.subplot(223)
plt.ylabel('TEST Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list2)

plt.subplot(224)
plt.ylabel('TEST Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list2, label = '2 hidden')













# TRAIN THE NETWORK, CHECK PREDICTIONS ON UNSEEN DATA AND ADD THEM TO PLOT

# Initialize the mlp
network= mlp.MLP(0.01, 1, 5)
network.learn(X, T)

#Setup 
i = 0
not_converged = 1
mse_list = []
mis_list = []
mse_list2 = []
mis_list2 = []


while(not_converged):
	network.continue_learning(1)

	#Predict mse on unseen test data
	output2 = network.predict(X2, T2)[0]
	mse2 = ((output2 - T2) ** 2).mean(axis=None)
	mse_list2.append(mse2)

	#Predict classification error on unseen test data
	actual2 = np.around(network.predict(X2, T2)[0]) 
	ideal2 = T2[0]
	error2 = actual2 - ideal2
	missclassifications2 = np.count_nonzero(error2)
	mis_list2.append(missclassifications2)

	#Calculate and append mean squared error
	output = network.predict(X, T)[0]
	mse = ((output - T) ** 2).mean(axis=None)
	mse_list.append(mse)

	#Count and append missclassifications
	actual = np.around(network.predict(X, T)[0]) 
	ideal = T[0]
	error = actual - ideal
	missclassifications = np.count_nonzero(error)
	mis_list.append(missclassifications)

	#Convergence check
	if (mse < 0.1):
		not_converged = 0

	i += 1

#Plot values
plt.figure(2)
plt.subplot(221)
plt.ylabel('TRAIN Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list)

plt.subplot(222)
plt.ylabel('TRAIN Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list, label = '5 hidden')

plt.subplot(223)
plt.ylabel('TEST Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list2)

plt.subplot(224)
plt.ylabel('TEST Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list2, label = '5 hidden')











# TRAIN THE NETWORK, CHECK PREDICTIONS ON UNSEEN DATA AND ADD THEM TO PLOT

# Initialize the mlp
network= mlp.MLP(0.01, 1, 10)
network.learn(X, T)

#Setup 
i = 0
not_converged = 1
mse_list = []
mis_list = []
mse_list2 = []
mis_list2 = []


while(not_converged):
	network.continue_learning(1)

	#Predict mse on unseen test data
	output2 = network.predict(X2, T2)[0]
	mse2 = ((output2 - T2) ** 2).mean(axis=None)
	mse_list2.append(mse2)

	#Predict classification error on unseen test data
	actual2 = np.around(network.predict(X2, T2)[0]) 
	ideal2 = T2[0]
	error2 = actual2 - ideal2
	missclassifications2 = np.count_nonzero(error2)
	mis_list2.append(missclassifications2)

	#Calculate and append mean squared error
	output = network.predict(X, T)[0]
	mse = ((output - T) ** 2).mean(axis=None)
	mse_list.append(mse)

	#Count and append missclassifications
	actual = np.around(network.predict(X, T)[0]) 
	ideal = T[0]
	error = actual - ideal
	missclassifications = np.count_nonzero(error)
	mis_list.append(missclassifications)

	#Convergence check
	if (mse < 0.1):
		not_converged = 0

	i += 1

#Plot values
plt.figure(2)
plt.subplot(221)
plt.ylabel('TRAIN Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list)

plt.subplot(222)
plt.ylabel('TRAIN Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list, label = '10 hidden')

plt.subplot(223)
plt.ylabel('TEST Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list2)

plt.subplot(224)
plt.ylabel('TEST Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list2, label = '10 hidden')












# TRAIN THE NETWORK, CHECK PREDICTIONS ON UNSEEN DATA AND ADD THEM TO PLOT

# Initialize the mlp
network= mlp.MLP(0.01, 1, 20)
network.learn(X, T)

#Setup 
i = 0
not_converged = 1
mse_list = []
mis_list = []
mse_list2 = []
mis_list2 = []


while(not_converged):
	network.continue_learning(1)

	#Predict mse on unseen test data
	output2 = network.predict(X2, T2)[0]
	mse2 = ((output2 - T2) ** 2).mean(axis=None)
	mse_list2.append(mse2)

	#Predict classification error on unseen test data
	actual2 = np.around(network.predict(X2, T2)[0]) 
	ideal2 = T2[0]
	error2 = actual2 - ideal2
	missclassifications2 = np.count_nonzero(error2)
	mis_list2.append(missclassifications2)

	#Calculate and append mean squared error
	output = network.predict(X, T)[0]
	mse = ((output - T) ** 2).mean(axis=None)
	mse_list.append(mse)

	#Count and append missclassifications
	actual = np.around(network.predict(X, T)[0]) 
	ideal = T[0]
	error = actual - ideal
	missclassifications = np.count_nonzero(error)
	mis_list.append(missclassifications)

	#Convergence check
	if (mse < 0.1):
		not_converged = 0

	i += 1

#Plot values
plt.figure(2)
plt.subplot(221)
plt.ylabel('TRAIN Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list)

plt.subplot(222)
plt.ylabel('TRAIN Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list, label = '20 hidden')

plt.subplot(223)
plt.ylabel('TEST Missclassifications')
plt.xlabel('epochs')
plt.plot(mis_list2)

plt.subplot(224)
plt.ylabel('TEST Mean Square Error')
plt.xlabel('epochs')
plt.plot(mse_list2, label = '20 hidden')














# SHOW THE PLOTS

legend = plt.legend(loc='upper right', shadow=True)
plt.show()
