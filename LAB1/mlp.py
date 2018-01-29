import numpy as np

def phi(x):
	return 2/(1 + np.exp(-x)) - 1

def phi_prime(x):
	return (1 + phi(x)) * (1 - phi(x)) / 2

class MLP(object):
	def __init__(self, eta, epochs, hidden):
		self.eta = eta
		self.epochs = epochs

		#Number of neurons in the hidden layer
		self.hidden = hidden

	def run(self, X, T):
		T = np.array(T)
		X = np.array(X)

		#Append bias vector to input 
		bias = np.ones(X.shape[1])
		X = np.vstack((X, bias))

		self.init_weights(X, T)
		self.learn(X,T)

	def init_weights(self, X, T):
		#V correspond to weights from X to hidden - the first layer
		V_rows = hidden
		V_cols = X.shape[0]

		#W correspond to weights from hidden  to hidden - the first layer.
		W_rows = T.shape[0]

		# +1 since we add input from the bias neuron
		W_cols = hidden + 1

		#Create a weight matrix for connections from input to hidden
		self.V = np.zeros(shape=(V_rows, V_cols))
		for i in range (V_rows):
			self.V[i] = np.random.normal(0,0.1,V_cols)

		#Create a weight matrix for connections from hidden to output 
		self.W = np.zeros(shape=(W_rows, W_cols))
		for i in range (W_rows):
			self.W[i] = np.random.normal(0,0.1,W_cols)

	def learn(self, X, T):
#FORWAD PASS
		#Calculate net input to hidden layer
		H_input = np.dot(self.V, X)

		#Apply the activation function
		for i in range(H_input.shape[0]):
			for j in range(H_input.shape[1]):
				H_input[i,j] = phi(H_input[i,j])

		#Create and append the bias vector to the input for the output layer
		bias = np.ones(H_input.shape[1])
		H_output = np.vstack((H_input, bias))

		#Calculate net input to output layer
		O_input = np.dot(self.W, H_output)

		#Apply the activation function
		for i in range(O_input.shape[0]):
				for j in range(O_input.shape[1]):
					O_input[i,j] = phi(O_input[i,j])

#BACKWARD PASS
		#Compute error on output layer
		error1 = O_input - T
		O_derivative = np.multiply((1 + O_input), (1 - O_input)) * 0.5
		delta_O = np.multiply(error1, O_derivative)

		#Compute error on hidden layer
		error2 = np.dot(np.transpose(self.W),delta_O)
		H_derivative = np.multiply((1 + H_output), (1 - H_output)) * 0.5
		delta_H = np.multiply(error2, H_derivative)

		#Remove the last row which contains the bias
		delta_H = np.delete(delta_H, delta_H.shape[0] -1, 0)

#WEIGHT UPDATE

eta = 0.01
epochs = 100
hidden = 3

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


	

MLP = MLP(eta, epochs, hidden)
MLP.run(in3, out3)



