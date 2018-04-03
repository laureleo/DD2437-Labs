import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM, MLPClassifier
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import losses
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
import math
from sklearn.pipeline import Pipeline
from sklearn import metrics, linear_model


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
    return model

def stacked_RBM(epochs, structure, verbose):
    print("Yeah!")


def autoencoder_loss_different_configuration():
    stacked_autoencoder(50, [], 1)
    plt.show()
    stacked_autoencoder(50, [150], 1)
    plt.show()
    stacked_autoencoder(50, [150, 100], 1)
    plt.show()
    stacked_autoencoder(50, [150, 100, 50], 1)
    plt.show()

#Creates an autoencoder with symmetrical hidden layers according to structure and plots the weights for each layer in the resulting network after training
def check_autoencoder_weight_representations(structure):
    sae = stacked_autoencoder(50, structure, 0)
    weights = sae.get_weights()
    print("Checking how the weight matrices are placed...")
    for i in range(11):
        print(weights[i].shape)
    for j in range(6):
        layer = j*2
        print("Looking at layer {}".format(layer))
        print("Weight matrix has shape {}".format(weights[layer].shape))
        print("Converting weights from input nodes to output nodes into images...")
        size = weights[layer].shape[0]
        dim = int(math.sqrt(weights[layer].shape[1]))
        imagecount = int(math.sqrt(size))
        plt.figure("Autoencoder weight image", figsize = (20, 20))
        for i in range(size):
            # display weights
            ax = plt.subplot(imagecount, imagecount, i + 1)
            plt.imshow(weights[layer][i].reshape(dim, dim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('weight_{}.png'.format(str(weights[layer].shape)))
        plt.show()

def create_DBN(structure, epochs):
    print("Creating a DBN")
    print("Setting up the {} RBMs".format(len(structure)))
    networks = []
    for i in range(len(structure)):
        rbm = BernoulliRBM(structure[i], batch_size = 100, learning_rate = 0.2,
                                      n_iter = epochs, verbose = True, random_state = 1)
        networks.append(('rbm{}'.format(i),rbm))
        print("\nCreated RBM with {} hidden".format(structure[i]))

    networks.append(('mlp', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',alpha=0.01)))
    model = Pipeline(networks)
    print("\nConnected all {} RBMs via a pipeline. Commencing training".format(len(structure)))
    model.fit(train_in, train_out)

    rbm = networks[0][1]

    plt.figure("DBN weight image 1", figsize = (20, 20))
    for i, comp in enumerate(rbm.components_):
        dim1 = int(math.sqrt(structure[0]))
        ax = plt.subplot(dim1, dim1, i+1)
        plt.imshow(comp.reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("DBN1weights")

    if(len(structure) > 1):
        dim2 = int(math.sqrt(structure[1]))
        rbm = networks[1][1]
        plt.figure("DBN weight image 2", figsize = (20, 20))
        for i, comp in enumerate(rbm.components_):
            ax = plt.subplot(dim2, dim2, i+1)
            plt.imshow(comp.reshape(dim1, dim1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("DBN2weights")

    if(len(structure) > 2):
        dim3 = int(math.sqrt(structure[2]))
        rbm = networks[2][1]
        plt.figure("DBN weight image 3", figsize = (20, 20))
        for i, comp in enumerate(rbm.components_):
            ax = plt.subplot(dim3, dim3, i+1)
            plt.imshow(comp.reshape(dim2, dim2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("DBN3weights")
    plt.show()



#check_autoencoder_weight_representations([169, 100, 49])
dbn = create_DBN([169, 100, 49], 20)
total_error = 0

