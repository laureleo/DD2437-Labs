import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def phi(x):
	return 2/(1 + np.exp(-x)) - 1

class MLP(object):

	def __init__(self, eta, epochs, hidden, draw=False, x_draw=[], y_draw=[]):
		self.eta = eta
		self.epochs = epochs
		self.hidden = hidden
        
      #Initialisation for the draw_now()
		self.draw=draw
		if draw:
			self.x_draw = x_draw
			self.y_draw = y_draw
			self.X_draw, self.Y_draw = np.meshgrid(x_draw, y_draw) 
		


	#Start to train a network from scratch. 
	def learn(self, X, T):
		self.setup(X, T)
		self.init_weights()
		for i in range(self.epochs):
			self.forward_pass()
			self.draw_now(i)
			self.backward_pass()
			self.update_weights()
	

	#Continues the learning process without resetting the weights
	def continue_learning(self, epochs):
		for i in range(epochs):
			self.forward_pass()
			self.backward_pass()
			self.update_weights()
	

	def setup(self, X, T):
		self.T = np.array(T)
		self.X = np.array(X)
		bias = np.ones(self.X.shape[1])
		self.X = np.vstack((self.X, bias))


	def init_weights(self):
		#V correspond to weights from X to hidden
		V_rows = self.hidden
		V_cols = self.X.shape[0]

		#W correspond to weights from hidden to output
		W_rows = self.T.shape[0]

		# +1 since we add input from the bias neuron
		W_cols = self.hidden + 1

		#Create a weight matrix for connections from input to hidden
		self.V = np.zeros(shape=(V_rows, V_cols))
		for i in range (V_rows):
			self.V[i] = np.random.normal(0,0.1,V_cols)

		#Create a weight matrix for connections from hidden to output 
		self.W = np.zeros(shape=(W_rows, W_cols))
		for i in range (W_rows):
			self.W[i] = np.random.normal(0,0.1,W_cols)


	def predict(self, X, T):
		self.setup(X, T)
		return self.O_output


	def forward_pass(self, X = []):
		#Calculate net input to hidden layer
		if X == []:
			X= self.X  
		else:
			X = np.append(X, np.ones((1, X.shape[1])), axis=0)
			
		self.H_input = np.dot(self.V, X)

		#Apply the activation function
		self.H_input = phi(self.H_input)

		#Create and append the bias vector to the input for the output layer
		bias = np.ones(self.H_input.shape[1])
		self.H_output = np.vstack((self.H_input, bias))

		#Calculate net input to output layer
		self.O_input = np.dot(self.W, self.H_output)

		#Apply the activation function
		self.O_input =  phi(self.O_input)
		self.O_output = self.O_input


	def backward_pass(self):
		#Compute error on output layer
		error1 = self.O_output- self.T
		self.O_derivative = np.multiply((1 + self.O_output), (1 - self.O_output)) * 0.5
		self.delta_O = np.multiply(error1, self.O_derivative)

		#Compute error on hidden layer
		error2 = np.dot(np.transpose(self.W), self.delta_O)
		self.H_derivative = np.multiply((1 + self.H_output), (1 - self.H_output)) * 0.5
		self.delta_H = np.multiply(error2, self.H_derivative)

		#Remove the last row which contains the bias
		self.delta_H = np.delete(self.delta_H, self.delta_H.shape[0] -1, 0)


	def update_weights(self):
		delta_W = -self.eta * np.dot(self.delta_O, np.transpose(self.H_output))
		delta_V = -self.eta * np.dot(self.delta_H, np.transpose(self.X))
		self.W += delta_W
		self.V += delta_V

	def view_w(self):
		return self.W

	def view_v(self):
		return self.V
    
	def draw_now(self, epoch):
		if self.draw:
			if epoch%10 == 0:    
				fig = plt.figure()
				ax = fig.gca(projection='3d')    
				zz = self.O_output.reshape(len(self.x_draw), len(self.y_draw)) #reshaping  result of the forward pass
				ax.plot_wireframe(self.X_draw,self.Y_draw,zz)
				title="step by step approximation, epoch="+str(epoch)
				ax.set_title(title)
				ax.set_zlim(bottom=-0.6, top=0.2) #arbitrary z-axis drawn from auto-scaled a-axis of the final prediction
				filename="./Images/Approximation"+str(epoch)+".png"
				plt.savefig(filename)
				plt.close(fig)
    
		
		
       
      

# Example use
# 
# 
# in1 = [
# [-2,4,-1],
# [4,1,-1],
# [1, 6, -1],
# [2, 4, -1],
# [6, 2, -1],
# ]
# 
# out1 = [[-1,-1, 1],[1,-1,1]]
# 
# eta = 0.01
# epochs = 1000
# hidden = 10
# 
# MLP = MLP(eta, epochs, hidden)
# MLP.learn(in1, out1)
# print(MLP.predict(in1, out1))
