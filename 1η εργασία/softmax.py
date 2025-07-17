import numpy as np

class Activation_Softmax:
    def __init__(self):
       self.outputs = []

    def forward(self, inputs):
        for input in inputs:
            exp_x = np.exp(input - np.max(input,  keepdims = True))
            self.outputs.append(exp_x / np.sum(exp_x))
        self.outputs = np.array(self.outputs)
        return self.outputs