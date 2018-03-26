import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model

def autoencode(epochs, verbose, performance):
    print("Running the autoencoder")

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

    #Set the dimensionality of the encoded input
    encoding_dim = 32  

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


    autoencoder.fit(train_in, train_in,
                    epochs=epochs,
                    batch_size=256,
                    shuffle=True,
                    verbose = verbose,
                    validation_data=(test_in, test_in))

# encode and decode some digits
# note that we take them from the *test* set
    encoded_imgs = encoder.predict(test_in)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure("Autoencoder", figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_in[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



#Use high to get the best performance. Use anything else to use what the labs tell us to use
autoencode(10, 0, "high")
