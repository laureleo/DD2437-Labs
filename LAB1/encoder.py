import mlp as mlp
import numpy as np

def gen_inputs(size):
	X = np.ones(shape=(size, size))
	X * -1
	for i in range(size):
		X[i,i] = -1
	return X

eta = 0.01 
epochs = 1
hidden = 3
encode = 8

X = gen_inputs(encode)
MLP = mlp.MLP(eta, epochs, hidden)
MLP.learn(X, X)

not_converged = 1
i = 0

while(not_converged):
	MLP.continue_learning(10)
	error = np.matrix.round(MLP.predict(X, X) - X) 
	if (np.all(error == 0)):
		print(i, " epochs to converge")
		not_converged = 0
	i+= 10

