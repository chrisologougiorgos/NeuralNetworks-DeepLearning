import numpy as np

class Activation_Sigmoid:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-np.clip(inputs, -709, 709)))
        return self.outputs

