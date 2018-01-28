import numpy as np

class Perceptron(object):

	def __init__(self, eta, epochs):
		self.eta = eta
		self.epochs = epochs

	def run(self, x, t):
		#np.array just ensures that our inputs are recognized as matrices by numpy
		T = np.array(t)
		X = np.array(x)

		#Fill a one-dimensional array (with the same number of cols as X) with 1:s. This is the bias vector
		bias = np.ones(X.shape[1])

		#Append the bias vector as the last row in the input matrix
		X = np.vstack((X, bias))

		#Initialize the weights, do the learning and return the final weight matrix
		self.init_weights(X, T)
		self.learn(X, T)
		return self.W

	def learn(self, X, T):

		#For each epoch
		for i in range(self.epochs):

			#Calculate net input to neuron
			net_input = np.dot(self.W, X)
			
			#For each element, apply the activation function
			for i in range(net_input.shape[0]):
				for j in range (net_input.shape[1]):
					if net_input[i,j] >= 0:
						net_input[i,j] =  1
					else:
						net_input[i,j] =  -1
					
			#Get the error
			error = net_input - T

			#Apply the delta rule
			delta = -1 * self.eta * np.dot(error, np.transpose(X))

			#Update weights 
			self.W += delta

	def init_weights(self, X, T):
		W_rows = output_dimensionality = T.shape[0]
		W_cols = input_dimensionality = X.shape[0]

		#Initialize a weight matrix with as many rows as the output dimensionality, as many cols as input dimensionality
		self.W = np.zeros(shape=(W_rows, W_cols))
		
		#Fill each row with values drawn randomly from a normal distribution with a stdev of 0.1
		for i in range (W_rows):
			self.W[i] = np.random.normal(0, 0.1, W_cols)

#Test running the perceptron
eta = 0.01
epochs = 10
in1 =[
	[1,1,-1,-1],
	[1,-1,1,-1]
	]

out1 =[[1,1,1,-1]]


in2 = [
	[-2,4,-1],
	[4,1,-1],
	[1, 6, -1],
	[2, 4, -1],
	[6, 2, -1],
	]

out2 = [[-1,-1, 1]]

in3 = [
	[-2,4,-1],
	[4,1,-1],
	[1, 6, -1],
	[2, 4, -1],
	[6, 2, -1],
	]

out3= [[-1,-1, 1],[1,-1,1]]

perceptron = Perceptron(eta, epochs)
print("Weights are")
print(perceptron.run(in1, out1))
