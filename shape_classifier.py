import numpy as np
from termcolor import colored

from neural_network import NeuralNetwork
from training_data import *


def train(model: NeuralNetwork):
    model.train(training_inputs, training_outputs, 100000)
    model.save()


categories = ["square", "circle"]

model = NeuralNetwork()
# train(model)
model.load()

prediction = model.predict(
        np.array([0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 1,
                  0, 0, 0, 0, 1, 0, 0, 1,
                  0, 0, 0, 0, 1, 0, 0, 1,
                  0, 0, 0, 0, 1, 1, 1, 1]))

print("Output: ", colored(prediction[0], "cyan"))
print("Prediction: ", colored(categories[int(round(prediction[0]))], "cyan"))
