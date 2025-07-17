import numpy as np
from sigmoid import Activation_Sigmoid
from relu import Activation_ReLU
from leaky_relu import Activation_LeakyReLU

class Layer:
    def __init__(self, n_neurons, n_inputs):
        # Xavier initialization για Sigmoid
        '''
        limit = np.sqrt(6 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_inputs))
        '''

        # He Initialization για ReLU / Leaky ReLu
        #'''
        limit = np.sqrt(2 / n_inputs)
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_inputs))
        #'''

        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        self.outputs = inputs @ self.weights.T + self.biases
        return self.outputs

    #Χρήση Leaky Relu
    #'''
    def activation(self):
        activation = Activation_LeakyReLU(alpha=0.01)  
        self.activated_outputs = activation.forward(self.outputs)
        return self.activated_outputs

    def backward(self, b, inputs, following_layer):
        activation = Activation_LeakyReLU(alpha=0.01)
        activation.forward(self.outputs)

        activation_derivative = np.where(self.outputs > 0, 1, activation.alpha)

        self.delta = (following_layer.delta @ following_layer.weights) * activation_derivative
        gradient_clip_value = 1.0
        self.delta = np.clip(self.delta, -gradient_clip_value, gradient_clip_value)

        weights_update = b * (self.delta.T @ inputs) / len(inputs)
        biases_update = b * np.mean(self.delta, axis=0)

        self.weights -= weights_update
        self.biases -= biases_update

    #'''

    # Χρήση Relu
    '''
    def activation(self):
        activation = Activation_ReLU()
        self.activated_outputs = activation.forward(self.outputs)
        return self.activated_outputs


    def backward(self, b, inputs, following_layer):
        activation = Activation_ReLU()
        activation.forward(self.outputs)

        activation_derivative = np.where(self.outputs > 0, 1, 0)

        self.delta = (following_layer.delta @ following_layer.weights) * activation_derivative
        gradient_clip_value = 1.0
        self.delta = np.clip(self.delta, -gradient_clip_value, gradient_clip_value)

        weights_update = b * (self.delta.T @ inputs) / len(inputs)
        biases_update = b * np.mean(self.delta, axis=0)

        self.weights -= weights_update
        self.biases -= biases_update
    '''

    # Χρήση sigmoid
    '''        
    def activation(self):
        activation = Activation_Sigmoid()
        self.activated_outputs = activation.forward(self.outputs)
        return self.activated_outputs

    def backward(self, b, inputs, following_layer):
        activation = Activation_Sigmoid()
        activation.forward(self.outputs)
        activation_derivative = activation.outputs * (1 - activation.outputs)
        self.delta = (following_layer.delta @ following_layer.weights) * activation_derivative

        # Υπολογισμός weights/biases updates για όλο το batch
        weights_update = b * (self.delta.T @ inputs) / len(inputs)
        biases_update = b * np.mean(self.delta, axis=0)

        # Ενημέρωση weights και biases
        self.weights -= weights_update
        self.biases -= biases_update
    '''


