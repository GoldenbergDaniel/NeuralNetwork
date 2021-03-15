import numpy as np
from termcolor import colored

from neural_network import NeuralNetwork
from dataset1 import *

model = NeuralNetwork(4)
model.train(training_inputs2, training_outputs2, 10000)
model.save()
# model.load()

prediction = model.predict(np.array(
                            [0, 0, 0, 0,
                             0, 0, 1, 0,
                             0, 1, 0, 1,
                             0, 0, 1, 0]))

print("Output: ", colored(prediction[0], "cyan"))
print("Prediction: ", colored(categories[int(round(prediction[0]))], "cyan"))
