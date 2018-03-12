import numpy as np


class SOM(object):
    def __init__(self, rows, cols, input_dim, neigh_size, eta, rnd_seed, decay_rate):
        self._rows = rows
        self._cols = cols
        self._rnd_gen = np.random.RandomState(rnd_seed)
        self._eta = eta
        self._neigh_size = neigh_size
        self._decay = decay_rate
        self._map = self._rnd_gen.rand(rows, cols, input_dim)
        self._activations = np.zeros((rows, cols))
        print("Created Self-Organizing Map with following attributes:\n Map dimensions = {}\n Input pattern dimensionality = {}\n Initial neighbourhood size = {}\n Learning rate = {}\n Randomization = {}\n Neighbourhood decay rate = {}\n".format((rows, cols), input_dim, neigh_size, eta, rnd_seed, decay_rate))

# Return the map.
    def get_map(self):
        return self._map

# Return the weights associated with the neuron at [row][col] in map
    def get_weight(self, row, col):
        return self._map[row, col, :]

# Set the weights associated with the neuron at [row][col] in map to be [new_weights]
    def set_weight(self, row, col, new_weight):
        self._map[row, col, :] =  new_weight
 
# Decays the size of the neighbourhood at a constant rate of [_decay] 
    def decay_neighbourhood(self):
        self._neigh_size = self._neigh_size - self._decay 

# Return the euclidean distance between the neuron at [row][col] and input pattern
    def get_distance(self, row, col,  pattern):
        neuron = self.get_weight(row, col)
        return np.linalg.norm(neuron - pattern)

# Returns how strongly each neuron in the map responds to an input pattern
    def get_activations(self, input_pattern):
        for i in range(self._rows):
            for j in range(self._cols):
                distance = self.get_distance(i, j, input_pattern)
                self._activations[i, j] = distance
        return(self._activations)

# Returns the coordinates of the neuron closest to the input
    def get_winner(self, input_pattern):
        self.get_activations(input_pattern)
        return np.unravel_index(self._activations.argmin(), self._activations.shape)

# Returns a list of tuples (neighbour coordinates) within neigh_size of the winner neuron. Uses manhattan distance i.e only counts distance as units travelled horizontally/vertically
    def get_neighbours(self, input_pattern):
        winner = self.get_winner(input_pattern)
        x = winner[0]
        y = winner[1]

        neigh_size = self._neigh_size 
        neighbours= []

        for i in range(x - neigh_size, x + neigh_size + 1):
            steps_left = np.absolute(np.absolute(x-i) - neigh_size)
            for j in range(y - steps_left, y + steps_left + 1):
                if(i >= 0 and j >= 0 and i < self._rows and j < self._cols):
                    neighbours.append((i,j))
        return(neighbours)

# Finds a winning neuron and moves it and its neighbours closer to the input that activated it. 
    def update_map(self, pattern):
        winner = self.get_winner(pattern)
        neighbours = self.get_neighbours(pattern)

        #print("\nlooking at pattern {}\nwinner is {} and the winners neighbours are".format(pattern, winner))
        #print(neighbours)

        for neighbour in neighbours:
            old_w = self.get_weight(neighbour[0], neighbour[1])
            new_w = old_w + self._eta * (pattern - old_w)
            self.set_weight(neighbour[0], neighbour[1], new_w)


# Trains the network for [epoch] epochs
    def train(self, patterns, epochs):
        for i in range(epochs):
            for pattern in patterns:
                self.update_map(pattern)
            self.decay_neighbourhood()

        print("Final neighbourhood size = {}".format(self._neigh_size))

# Returns an array of winners corresponding to the patterns received. 
    def get_winners(self, patterns):
        winner_list = []
        for pattern in patterns:
            winner = self.get_winner(pattern)
            winner_list.append(winner)
            
        return winner_list








""" 
Topological ordering of animal species
Inputs are 32 rows of 84 attributes, each row corresponding to one animal
Output nodes should be 100
Learning rate should be 0.2
Initial neigbourhood size should be 50 and end up around 1
Output should be one-dimensional
neighbourhood should be one-dimensional
train for approx 20 epochs
"""

animals = np.loadtxt("./data/animals.dat", dtype='i', delimiter=',')
animals = np.ndarray.reshape(animals, 32, 84)

animal_names= np.loadtxt("./data/animalnames.txt", dtype='string', delimiter='\n')

som = SOM(1,100,84,50,0.2,2,2)
som.train(animals, 23)
result = som.get_winners(animals)

result2 = []

for i in range(32):
    result2.append((result[i][1], animal_names[i]))

print("Animals more closely related should have appear more closely in the following list")
for i in range(100):
    for j in range(32):
        if (result2[j][0] == i):
            print result2[j]
        


