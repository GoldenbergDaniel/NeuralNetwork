import sys

import numpy as np
from termcolor import colored


class NeuralNetwork:
    def __init__(self):
        self.weights = 2 * np.random.random((3, 1)) - 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, y):
        return y * (1 - y)
    def train(self, training_inputs, training_outputs, iterations):
        for _ in range(iterations):
            output = self.predict(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustments
    def predict(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output
    def save(self):
        np.save("model.npy", self.weights)
    def load(self):
        self.weights = np.load("model.npy")


def train(model: NeuralNetwork):
    training_inputs = np.array([
    [0, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    model.train(training_inputs, training_outputs, 10000)
    model.save()


model = NeuralNetwork()
train(model)

X = int(sys.argv[1])
Y = int(sys.argv[2])
Z = int(sys.argv[3])

prediction = model.predict(np.array([X, Y, Z]))

print("Output: ", colored(prediction[0], "cyan"))
print("Prediction: ", colored(int(round(prediction[0])), "cyan"))
