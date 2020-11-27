import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = 2 * np.random.random((64, 4)) - 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, y):
        return y - y**2
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
        np.save("shape_classifier.npy", self.weights)
    def load(self):
        self.weights = np.load("shape_classifier.npy")
        np.save("model.npy", self.weights)
    def load(self):
        self.weights = np.load("model.npy")
