import mlp as mlp
import numpy as np

def gen_inputs(size):
	X = np.ones(shape=(size, size))
	X * -1
	for i in range(size):
		X[i,i] = -1
	return X

eta = 0.01 
epochs = 10000
hidden = 8
encode= 8


X = gen_inputs(encode)
MLP = mlp.MLP(eta, epochs, hidden)
MLP.learn(X, X)
MLP.predict(X, X)

