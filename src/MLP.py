# Week 4 - Tutorial 5 - Multi Layer Perceptron
# Task 4
# Import the libraries
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from random import randint


# Get iris dataset
def iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split the data into training and testing
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0, 1000))
    return X_train, X_test, y_train, y_test

# Train the MLP
def train_MLP(X_train, X_test, y_train, y_test, hidden_layer=(10,), iterations=2000):
    # Create the model
    model = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=iterations, activation='relu', solver='adam', random_state=1)

    # Fit the model
    model.fit(X_train, y_train)

    # Print the results
    print("Test score accuracy = {}".format(model.score(X_test, y_test)))
    print("Training score accuracy = {}".format(model.score(X_train, y_train)))
    # print("Predicted Clasees = {}".format(model.predict(X_test)))
    print("Confusion Matrix = {}".format(confusion_matrix(y_test, model.predict(X_test))))


def main():
    # Get the data
    X, y = iris_data()
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Train the MLP
    train_MLP(X_train, X_test, y_train, y_test)

# Run the main function
if __name__ == "__main__":
    main()