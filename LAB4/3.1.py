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



def autoencode(epochs, verbose, performance, hidden):
    print("Running the autoencoder")
    #Set the dimensionality of the encoded input
    encoding_dim = hidden 

    #Placeholders. Dunno why these are necessary, the Keras guide recommends it
    input_img = Input(shape=(784,))
    encoded_input = Input(shape=(encoding_dim,))

    #Layer that encodes the input (from 784 dims to 32)
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    #Layer that decodes the input (from 32 dims to 784)
    decoded = Dense(784, activation='sigmoid')(encoded)

    # Model that maps an input to itself
    autoencoder = Model(input_img, decoded)

    # Model that maps an input to its encoded version
    encoder = Model(input_img, encoded)

    # Model that maps an input to its decoded version 
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    if(performance == "high"):
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    else:
        autoencoder.compile(optimizer='sgd', loss='mse')

    # Setup early stopping
    earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5,
                                              verbose=0, mode='auto')
    callback_list = [earlystop]


    history = autoencoder.fit(train_in, train_in,
                    epochs=epochs,
                    batch_size=256,
                    shuffle=True,
                    verbose = verbose,
                    validation_data=(test_in, test_in),
                    callbacks = callback_list)
    
    plt.figure("Autoencoder Loss")
    plt.plot(history.history['loss'])
    return autoencoder

# encode and decode some digits
# note that we take them from the *test* set
    encoded_imgs = encoder.predict(test_in)
    decoded_imgs = decoder.predict(encoded_imgs)

    samples = [18, 3, 7, 0, 2, 1,15 , 8, 6 ,5 ]
    n = 10  # how many digits we will display
    plt.figure("Autoencoder", figsize=(20, 4))
    for i in range(n):
        index = samples[i]
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_in[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



def rbm(epochs, hidden, eta, graph_error_epoch_relation):
    print("Running the rbm...")

    if(graph_error_epoch_relation):
        print("Graphing average error as a function of epoch...")
        error_list = []
        for i in range(epochs):
            rbm = BernoulliRBM(n_components=hidden, learning_rate=eta, batch_size=100, n_iter=i, verbose = True, random_state = 1)
            rbm.fit(train_in)

            total_error = 0
            for image in train_in:
                reconstruction = rbm.gibbs(image).astype(int)
                error = mean_squared_error(image, reconstruction)
                total_error = total_error + error

            error_list.append(total_error/8000)
        print(error_list)
        plt.figure("Epoch-Loss relation in RBM")
        plt.plot(error_list)


    print("Creating reconstructed images, using test data...")
    rbm = BernoulliRBM(n_components=hidden, learning_rate=eta, batch_size=100, n_iter=epochs, verbose = True, random_state = 1)
    rbm.fit(train_in)


    plt.figure("RBM mnist digits", figsize = (20, 4))
    samples = [18, 3, 7, 0, 2, 1,15 , 8, 6 ,5 ]
    i = 0
    for index in samples:
        image = test_in[index]
        reconstruction = rbm.gibbs(image).astype(int)

        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(image.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(reconstruction.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        i = i + 1
    plt.show()
    return rbm



#Using sgd and mse
#autoencode(50, 0, "high", 150)

#Using adadelta and binary cross entropy
#autoencode(50, 0, "low", 150)

# Show the weights for output neurons in the autoencoder as images
ae = autoencode(50, 0, "high", 100)
weights = ae.get_weights()[2]
print(weights)
#size = weights.shape[0]
#plt.figure("Autoencoder weight image", figsize = (20, 20))
#for i in range(size):
#    # display weights
#    ax = plt.subplot(10, 10, i + 1)
#    plt.imshow(weights[i].reshape(28, 28))
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.savefig("AEweights100.png")
#plt.show()
#

# Show the weights for output neurons in the rbm
rbm = rbm(50, 100, 0.2, 0)
plt.figure("RBM weight image", figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("H100.png")

plt.show()
