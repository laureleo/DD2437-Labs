import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import losses
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error


train_in = np.loadtxt("./data/bindigit_trn.csv", dtype='i', delimiter=',')
train_in = np.ndarray.reshape(train_in, 8000, 784)
print("Loaded training input patterns...")

train_out= np.loadtxt("./data/targetdigit_trn.csv", dtype='i', delimiter=',')
train_out= np.ndarray.reshape(train_out, 8000, 1)
print("Loaded training output patterns...")

test_in= np.loadtxt("./data/bindigit_tst.csv", dtype='i', delimiter=',')
test_in = np.ndarray.reshape(test_in, 2000, 784)
print("Loaded test input patterns...")

test_out= np.loadtxt("./data/targetdigit_tst.csv", dtype='i', delimiter=',')
test_ou= np.ndarray.reshape(test_out, 2000, 1)
print("Loaded test output patterns...")

def stacked_autoencoder(epochs, structure, verbose):
    print("\nCreated autoencoder\n\nEpochs = {}\nStructure = {}\nVerbose = {}".format(epochs, structure, verbose))

    input_layer = Input(shape=(784,))
    layer = input_layer

#Starting from the input layer, add on hidden layers according to structure, connecting each to the preceeding layer
    for hidden_layer in structure:
        layer = Dense(hidden_layer, activation = 'relu')(layer)

    if(len(structure) >0):
        structure.pop(-1)
    
#Starting from the final encoding layer, apply decoding layers in reverse
    for hidden_layer in reversed(structure):
        layer = Dense(hidden_layer, activation = 'relu')(layer)
    output_layer = Dense(784, activation = 'relu')(layer)

    model = Model(input_layer, output_layer)
    model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

    history = model.fit(train_in, train_in,
                   epochs=epochs,
                   batch_size=256,
                   shuffle=True,
                   validation_data = (test_in, test_in),
                   verbose=verbose)
    plt.figure("Stacked autoencoder loss")
    plt.plot(history.history['loss'])
    plt.show()


#Performance on stacked autoencoder classification with different amount of hidden layer
#stacked_autoencoder(50, [150], 1)
#stacked_autoencoder(50, [150, 100], 1)
#stacked_autoencoder(50, [150, 100, 50], 1)


