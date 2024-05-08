class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # adds layer to network
    def addLayer(self, Layer):
        self.layers.append(Layer)

    # add loss function to network
    def addLoss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # prediction on test data
    def predict(self, input_data):
        samples = len(input_data)
        results = []

        for i in range(samples):

            # forward propogation
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propogation(output)
            results.append(output)

        return results
    
    # training the model
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0

            for j in range(samples):
                output = x_train[j]

                # forward pass
                for layer in self.layers:
                    output = layer.forward_propogation(output)

                # loss calculation
                err += self.loss(y_train[j], output)

                # backward pass
                error = self.loss_prime(y_train[j], output)
                
                for layer in reversed(self.layers):
                    error = layer.backward_propogation(error, learning_rate)
            
            # calculate avg error across all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

