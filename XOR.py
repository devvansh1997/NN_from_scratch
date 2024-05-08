import numpy as np

from nn import FCLayer, ActivationLayer
from loss_functions import mse, mse_prime
from network import Network
from activation_functions import tanh, tanh_prime

# XOR EXAMPLE
# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# intializing model + building model
nn_net = Network()
nn_net.addLayer(FCLayer(2,3))
nn_net.addLayer(ActivationLayer(tanh, tanh_prime))
nn_net.addLayer(FCLayer(3,1))
nn_net.addLayer(ActivationLayer(tanh, tanh_prime))

# training model
nn_net.addLoss(mse, mse_prime)
nn_net.fit(x_train = x_train, y_train = y_train, epochs = 1000, learning_rate= 0.1)

# test
out = nn_net.predict(x_train)
print(out)