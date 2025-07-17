import numpy as np

class Activation_ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.outputs <= 0] = 0
