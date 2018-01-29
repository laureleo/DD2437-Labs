import numpy as np

def phi(x):
	return 2/(1 + np.exp(-x)) - 1

class MLP(object):
	def __init__(self, eta, epochs, hidden):
		self.eta = eta
		self.epochs = epochs
		self.hidden = hidden

	def predict(self, X, T):
		T = np.array(T)
		X = np.array(X)

		#Append bias vector to input 
		bias = np.ones(X.shape[1])
		X = np.vstack((X, bias))
		actual = self.forward_pass(X, T)
		print("actual")
		print(actual)
		print("ideal")
		print(T)

	def learn(self, X, T):
		T = np.array(T)
		X = np.array(X)

		#Append bias vector to input 
		bias = np.ones(X.shape[1])
		X = np.vstack((X, bias))

		#Initialize weights
		self.init_weights(X, T)
		
		for i in range(epochs):
			x = self.forward_pass(X, T)
			self.backward_pass(T)
			self.update_weights(X)

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

	def forward_pass(self, X, T):
		#Calculate net input to hidden layer
		self.H_input = np.dot(self.V, X)

		#Apply the activation function
		for i in range(self.H_input.shape[0]):
			for j in range(self.H_input.shape[1]):
				self.H_input[i,j] = phi(self.H_input[i,j])

		#Create and append the bias vector to the input for the output layer
		bias = np.ones(self.H_input.shape[1])
		self.H_output = np.vstack((self.H_input, bias))

		#Calculate net input to output layer
		self.O_input = np.dot(self.W, self.H_output)

		#Apply the activation function
		for i in range(self.O_input.shape[0]):
			for j in range(self.O_input.shape[1]):
				self.O_input[i,j] = phi(self.O_input[i,j])

		actual = self.O_input
		return actual

	def backward_pass(self, T):
		#Compute error on output layer
		error1 = self.O_input - T
		self.O_derivative = np.multiply((1 + self.O_input), (1 - self.O_input)) * 0.5
		self.delta_O = np.multiply(error1, self.O_derivative)

		#Compute error on hidden layer
		error2 = np.dot(np.transpose(self.W), self.delta_O)
		self.H_derivative = np.multiply((1 + self.H_output), (1 - self.H_output)) * 0.5
		self.delta_H = np.multiply(error2, self.H_derivative)

		#Remove the last row which contains the bias
		self.delta_H = np.delete(self.delta_H, self.delta_H.shape[0] -1, 0)

	def update_weights(self, X):
		print(self.W)
		self.W = eta * self.W -(1 - eta) * np.dot(self.delta_O, np.transpose(self.H_output))
		print(self.W)
		self.V = eta * self.V -(1 - eta) * np.dot(self.delta_H, np.transpose(X))

eta = 0.9
epochs = 10
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


in4 = [
	[1],
	[1]
	]
out4 = [[1]]
	

MLP = MLP(eta, epochs, hidden)
MLP.learn(in1, out1)
MLP.predict(in4, out4)



