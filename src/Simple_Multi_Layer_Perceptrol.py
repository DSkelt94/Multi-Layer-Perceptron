# Week 4 Neural Networks - Tutorial 5 - Simple Multi Layer Perceptron
# Task 3: Implement a Multi Layer Perceptron with 2 hidden layers and 2 neurons in each layer.
# Importing the libraries
from sklearn.neural_network import MLPClassifier

def data():
    return [[0, 0], [0, 1], [1, 0], [1, 1]]

# XOR function
def perceptronXOR(data):
    return list(map(lambda x: 1 if x[0] != x[1] else 0, data))

# AND function
def perceptronAND(data):
    return list(map(lambda x: 1 if x[0] == 1 and x[1] == 1 else 0, data))

# NAND function
def perceptronNAND(data):
    return list(map(lambda x: 1 if x[0] == 0 and x[1] == 0 else 0, data))

# Training the neuron
def train_neuron(dataX, dataY):
    activations = ['logistic','identity', 'tanh', 'relu']
    for i in activations:
        model = MLPClassifier(activation=i)
        model.fit(dataX, dataY) # Fitting the data
        print("Activation function: {}".format(i))
        print(model.score(dataX, dataY)) # Getting the score
        print(model.predict(dataX)) # Getting the prediction

def main():
    X = data() # Getting the data
    Y = perceptronXOR(X) # Getting the XOR function
    #Y = perceptronAND(X) # Getting the AND function

    train_neuron(X, Y) # Training the neuron

if __name__ == "__main__":
    main()