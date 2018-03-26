import numpy as np
from sklearn.neural_network import BernoulliRBM

train_in = np.loadtxt("./data/bindigit_trn.csv", dtype='i', delimiter=',')
train_in = np.ndarray.reshape(train_in, 784, 8000)
print("Loaded training input patterns...")

train_out= np.loadtxt("./data/targetdigit_trn.csv", dtype='i', delimiter=',')
train_out= np.ndarray.reshape(train_out, 1, 8000)
print("Loaded training output patterns...")

test_in= np.loadtxt("./data/bindigit_tst.csv", dtype='i', delimiter=',')
test_in = np.ndarray.reshape(test_in, 784, 2000)
print("Loaded test input patterns...")

test_out= np.loadtxt("./data/targetdigit_tst.csv", dtype='i', delimiter=',')
test_ou= np.ndarray.reshape(test_out, 1, 2000)
print("Loaded test output patterns...")

ETA = 0.1
HIDDEN = 10
EPOCHS = 100
model = BernoulliRBM(
        batch_size=10,
        learning_rate=ETA,
        n_components=HIDDEN,
        n_iter=EPOCHS,
        random_state=1,
        verbose=1)

print("\nCreated RMB with:\nETA {}\nHIDDEN_SIZE {}\nEPOCHS {}\n".format(ETA,HIDDEN,EPOCHS))
model.fit(test_in)
