import numpy as np

class Perceptron(object):

	def __init__(self, eta, epochs):
		self.eta = eta
		self.epochs = epochs


	def run_pct(self, X, T):
		self.setup(X, T)
		self.init_weights()
		self.learn()


	def setup(self, X, T):
		#Convert input into a format numpy can deal with
		self.T = np.array(T)
		self.X = np.array(X)

		#Fill a one-dimensional array (with the same number of cols as X) with 1:s. This is the bias vector
		bias = np.ones(self.X.shape[1])

		#Append the bias vector as the last row in the input matrix
		self.X = np.vstack((self.X, bias))


	def init_weights(self):
		W_rows = output_dimensionality = self.T.shape[0]
		W_cols = input_dimensionality = self.X.shape[0]

		#Initialize a weight matrix with as many rows as the output dimensionality, as many cols as input dimensionality
		self.W = np.zeros(shape=(W_rows, W_cols))
		
		#Fill each row with values drawn randomly from a normal distribution with a stdev of 0.1
		for i in range (W_rows):
			self.W[i] = np.random.normal(0, 0.1, W_cols)

	
	def predict(self, X, T):
		self.setup(X, T)
		net_input = np.dot(self.W, self.X)
		for i in range(net_input.shape[0]):
			for j in range (net_input.shape[1]):
				if net_input[i,j] >= 0:
					net_input[i,j] =  1
				else:
					net_input[i,j] =  -1
		return(net_input)
		

	def learn(self):
		#For each epoch
		for i in range(self.epochs):

			#Calculate net input to neuron
			net_input = np.dot(self.W, self.X)
			
			#For each element, apply the activation function
			for i in range(net_input.shape[0]):
				for j in range (net_input.shape[1]):
					if net_input[i,j] >= 0:
						net_input[i,j] =  1
					else:
						net_input[i,j] =  -1
					
			#Get the error
			error = net_input - self.T

			#Apply the perceptron learning rule
			update = -1 * self.eta * np.dot(error, np.transpose(self.X))

			#Update weights 
			self.W += update 


#Example use
#
#in1 =[
#	[1,1,-1,-1],
#	[1,-1,1,-1]
#	]
#
#out1 =[[1,1,1,-1]]
#
#eta = 0.01
#epochs = 100
#pct = Perceptron(eta, epochs)
#pct.run_pct(in1, out1)
#print(pct.predict(in1, out1))
