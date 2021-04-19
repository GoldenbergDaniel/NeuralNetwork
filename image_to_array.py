import numpy as np
from PIL import Image
import os

path = "./dataset"

traning_inputs_8x8 = np.array()
traning_inputs_16x16 = np.array([[]])

traning_outputs_8x8 = np.array()
traning_outputs_16x16 = np.array([[]])

for folder in os.listdir(path):
    for img in os.listdir(os.path.join(path, folder)):
        img = Image.open(img)
        if "8x8" in folder:
            traning_inputs_8x8 = np.append(traning_inputs_8x8, img)
        if "16x16" in folder:
            traning_inputs_16x16 = np.append(traning_inputs_16x16, img)

np.save("traning_inputs_8x8.npy", traning_inputs_8x8)
np.save("traning_inputs_16x16.npy", traning_inputs_16x16)
