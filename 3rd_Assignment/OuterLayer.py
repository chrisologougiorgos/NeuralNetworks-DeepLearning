import numpy as np
from Softmax import Activation_Softmax

class OuterLayer:
    def __init__(self, n_neurons, n_inputs):
        self.biases = np.zeros(n_neurons)
        limit = np.sqrt(6 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_inputs))


    def forward(self, inputs):
        self.outputs = inputs @ self.weights.T + self.biases
        return self.outputs


    def activation(self):
        activation = Activation_Softmax()
        self.activated_outputs = activation.forward(self.outputs)
        return self.activated_outputs


    def backward(self, b, inputs, d):
        self.delta = self.activated_outputs - d
        weights_update = b * (self.delta.T @ inputs) / len(inputs)
        biases_update = b * np.mean(self.delta, axis=0)
        self.weights = self.weights - weights_update
        self.biases = self.biases - biases_update

    def train_least_squares(self, inputs, d):
        inputs_with_bias = np.hstack([inputs, np.ones((inputs.shape[0], 1))])

        pseudo_inverse = np.linalg.pinv(inputs_with_bias)

        weights_bias = pseudo_inverse @ d  # (M+1 x K)

        self.weights = weights_bias[:-1, :].T
        self.biases = weights_bias[-1, :]

