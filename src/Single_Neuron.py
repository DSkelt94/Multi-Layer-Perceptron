# Week 4 Neural Networks - Tutorial 5 - Single Perceptron 
# Task 1 - Implement a Single Perceptron with 2 inputs and 1 output.
# Importing the libraries
from sklearn.linear_model import Perceptron
from sklearn import datasets

# Getting the data
def data():
    return [[0, 0], [0, 1], [1, 0], [1, 1]]

# XOR function
def perceptronXOR(data):
    return list(map(lambda x: 1 if x[0] != x[1] else 0, data))

# AND function
def perceptronAND(data):
    return list(map(lambda x: 1 if x[0] == 1 and x[1] == 1 else 0, data))

# Training the neuron
def train_neuron(dataX, dataY):
    model = Perceptron() # Creating the model
    model.fit(dataX, dataY) # Fitting the data

    print(model.score(dataX, dataY)) # Getting the score
    print(model.predict(dataX)) # Getting the prediction

# Main function
def main():
    X = data() # Getting the data
    y = perceptronXOR(X) # Getting the XOR function
    #y = perceptronAND(X) # Getting the AND function

    train_neuron(X, y) # Training the neuron

if __name__ == "__main__":
    main()