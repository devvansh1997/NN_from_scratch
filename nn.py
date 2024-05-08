import numpy as np

# BASE CLASS: Layer:
# our base class layer has functions and params which every layer will inherit

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # given X (input) for a layer -> calculate Y (output)
    def forward_propogation(self, input):
        raise NotImplementedError
    
    # given dE/dY (derivative of error w.r.t output) -> calculate dE/dX (derivative of error w.r.t input)
    def backward_propogation(self, output_error, learning_rate):
        raise NotImplementedError
    
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1,output_size) - 0.5

    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propogation(self, output_error, learning_rate):
        # calculate dE/dX
        input_error = np.dot(output_error, self.weights.T)
        # calculate dE/dW
        weights_error = np.dot(self.input.T, output_error)
        # calculate dE/dB
        # bias_error = output_error

        # update params
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error

        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propogation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error