import numpy as np

# Tuto
# simple-nn-with-python-multi-layer-perceptron

class MLP:
    class Layer:
        def __init__(self, number_of_nodes, activation_function):
            self.nodes = number_of_nodes
            self.f = activation_function

        def f(self, x):
            if self.f == "sigmoid"
                return self.sigmoid(x)
            else:
                raise TypeError(f"Error: unknown activation function: {self.f}")

    def __init__(self, seed=10, activation_function="sigmoid", alpha=0.1):
        self.w = []
        self.b = []
        self.hidden_layers = []
        self.mu = []
        self.seed = seed
        self.f = activation_function
        self.alpha = alpha

        np.random.seed(self.seed)

    # Activations:
    def sigmoid(self):
        # perform sigmoid
        pass

    def activation_f(self, x):
        if self.f == "sigmoid"
            return self.sigmoid(x)
        else:
            raise TypeError(f"Error: unknown activation function: {self.f}")
        
    def add_layer(self, shape=(4, "sigmoid")):
        self.hidden_layers.append(self.Layer(shape=shape))

    def fit(self, X, Y):
        # fit
        # 
        # init of w and b for each layer
        for i in range(0, len(self.hidden_layers)):


        # feed foward

        # backprop

        pass

    def predict(self, X):
        # predict 
        pass

    def describe(self):
        pass

    def feed_forward(self, w, b, x, f):
        # compute node
        pass

    def back_propagation(self, lol):
        # back prop
        pass