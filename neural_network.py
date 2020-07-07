import sys

import numpy as np
from termcolor import colored

class NeuralNetwork:
    def __init__(self):
        self.weights = 3 * np.random.random((4, 1)) - 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
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


if __name__ == "__main__":
    model = NeuralNetwork()

    training_inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    model.train(training_inputs, training_outputs, 100000)

    A = int(sys.argv[1])
    B = int(sys.argv[2])
    C = int(sys.argv[3])

    print("Prediction: ", colored(model.predict(np.array([A, B, C])), "cyan"))
