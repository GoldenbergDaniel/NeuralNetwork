import numpy as np
from termcolor import colored

from neural_network import NeuralNetwork
from dataset1 import *

model = NeuralNetwork(8)
model.train(training_inputs1, training_outputs1, 10000)
model.save()
# model.load()

prediction = model.predict(np.array(
                           [0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 1, 1, 1, 1, 0, 0,
                            0, 0, 1, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 1, 0, 0,
                            0, 0, 1, 1, 1, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0]))

print("Output: ", colored(prediction[0], "cyan"))
print("Prediction: ", colored(categories[int(round(prediction[0]))], "cyan"))
