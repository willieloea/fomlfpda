class Perceptron:
    def __init__(self, weights, bias, activation_function):
        self.weights = list(weights)
        self.bias = float(bias)
        self.activation_function = activation_function

    def predict(self, input_signals):
        z = sum([x * y for x,y in zip(self.weights, input_signals)]) + self.bias
        return self.activation_function(z)

    def train(self, training_set, eta, epochs):
        assert len(training_set)

        # Training loop
        for epoch in range(epochs):
            errors_in_epoch = 0
            for x,y in training_set:
                error = y - self.predict(x)
                if error != 0:
                    for i in range(len(x)):
                        self.weights[i] = self.weights[i] + eta * error * x[i]
                    self.bias = self.bias + eta * error
                    errors_in_epoch += 1

            if errors_in_epoch == 0:
                return epoch

    """Activation Functions"""
    @staticmethod
    def step(z):
        return 1 if z >= 0 else 0


if __name__ == '__main__':
    D_1 = [
        [(0,0), 0],
        [(0,1), 0],
        [(1,0), 0],
        [(1,1), 1]
    ]
    andp = Perceptron([0,0], 0, Perceptron.step)
    nr_epochs = andp.train(D_1, 1, 1000)
    print("AND Perceptron")
    print("Weights:", andp.weights)
    print("Bias:", andp.bias)
    print(andp.predict([0,0]))
    print(andp.predict([0,1]))
    print(andp.predict([1,0]))
    print(andp.predict([1,1]))
    print("Number of epochs:", nr_epochs)

    D_2 = [
        [(0,0), 0],
        [(0,1), 1],
        [(1,0), 1],
        [(1,1), 1]
    ]
    orp = Perceptron([0,0], 0, Perceptron.step)
    nr_epochs = orp.train(D_2, 1, 1000)
    print("\nOR Perceptron")
    print("Weights:", orp.weights)
    print("Bias:", orp.bias)
    print(orp.predict([0,0]))
    print(orp.predict([0,1]))
    print(orp.predict([1,0]))
    print(orp.predict([1,1]))
    print("Number of epochs:", nr_epochs)

