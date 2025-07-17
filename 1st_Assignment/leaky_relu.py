import numpy as np

class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):  # alpha είναι ο παράγοντας διαρροής
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha