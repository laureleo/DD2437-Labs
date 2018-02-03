import mlp as mlp
import numpy as np

#Generate inputs
def gen_inputs(size):
	X = np.ones(shape=(size, size))
	X = -X
	for i in range(size):
		X[i,i] = 1
	return X


#Create an encoding, print the weights
def encode (eta, hidden, inputs, epochs):
	#Generate inputs
	X = gen_inputs(inputs)

	#Create the MLP
	MLP = mlp.MLP(eta, 1, hidden)

	#Initialize the MLP
	MLP.learn(X, X)

	#Stop condition
	not_converged = 1

	#counter until convergence
	i = 0

	while(not_converged):
		#Continue learning wit
		MLP.continue_learning(epochs)
		error = np.matrix.round(MLP.predict(X, X) - X) 
		if (np.all(error == 0)):
			not_converged = 0
		i+= epochs
	return MLP, i


#Average amount of epochs until convergence, latest run: 5003
def get_avg():
	summer = 0
	for i in range(1000):		
		x = encode(0.01, 3, 8, 10)
		summer += x[1]

	print("Average amount of epochs until convergence = ", summer/1000);

#Return the weights of the first layer as signed ones
def analyse():
	x = encode(0.01, 3, 8, 10)
	mlp = x[0]
	matrix = mlp.view_v()
	X = gen_inputs(8)
	bias = np.ones(X.shape[1])
	X = np.vstack((X, bias))
	activations = np.dot(matrix, X)
	show = np.transpose(activations)
	for i in range(show.shape[0]):
		for j in range(show.shape[1]):
			if(show[i,j] < 0):
				show[i,j]= -1
			else:
			 	show[i,j] = 1
	print("Inputs to each neuron is showed here in each column")
	print("Note that it looks like a truth table")
	print(show)

analyse()


'''
	After training the set of weights is such that each input causes a unique combination of hidden neurons to fire.
	Given that we have 3 hidden neurons we have 2^3 = 8 possible firing patterns, each of which correspond to one of the input patterns

'''
