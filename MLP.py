import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, accuracy_score
import math
import pickle


def _sigmoid(x, der=False):
    if der is False:
        r = 1 / (1 + np.exp(-x))
    else:
        r = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    return r

def _softmax(x, der=False):
    # print(x.shape)
    if x.shape[0] == 1:
        return _sigmoid(x, der=der)
    r = x.copy()
    if der is False:
        for i in range(0, len(x)):
            r[i] = np.exp(x[i]) / (np.exp(x).sum())
        # print(r)
    # print("ici")
    return r

class MultiLayerPerceptron:
    class Layer:
        def __init__(self, size=1, act_f="sigmoid", label="hidden_layer"):
            self.size = size
            self.f = act_f
            self.act_f = self.get_act_f()
            self.label=label

        def act_f(self, x, **kwargs):
            return self.get_act_f()(x, **kwargs)

        def get_act_f(self, string=False):
            if string is True:
                return self.f
            if self.f == "sigmoid":
                return _sigmoid
            elif self.f == "softmax":
                return _softmax
            else:
                raise TypeError(f"Error: unknown activation function: {self.f}")
        
        def size():
            return self.size

    def __init__(self, seed=10, alpha=0.1, layers=[], treshold=0.5):
        self.w = []
        self.b = []
        self.layers = []
        self.metrics = {'losses': [], 'scores': [], 'binary_cross_entropy': []}
        self.seed = seed
        self.act_f = []
        self.alpha = alpha
        self.treshold = treshold
        self.normalization = {}

        np.random.seed(self.seed)

        if len(layers) > 0:
            for i in range(0, len(layers)):
                self.add_layer(size=layers[i][0], act_f=layers[i][1], label=layers[i][2])

    # Activations:

    def add_layer(self, size=1, act_f="sigmoid", label="hidden_layer"):
        self.layers.append(self.Layer(size=size, act_f=act_f, label=label))
    
    def feed_forward(self, w, b, act_f, x):
        return act_f(np.dot(w, x) + b)

    def back_propagation(self, x, y):
        W = []
        B = []
        delta = []

        delta.insert(0, (self.z[len(self.z)-1] - y) * self.act_f[len(self.z)-1](self.z[len(self.z)-1], der=True))
        
        # Bqck propagation
        for i in range(0, len(self.z) - 1):
            delta.insert(0, np.dot(delta[0], self.w[len(self.z) - 1 - i]) * self.act_f[len(self.z) - 2 - i](self.z[len(self.z) - 2 - i], der=True))
        
        for i in range(0, len(delta)):
            delta[i] = delta[i] / self.X.shape[0]
   
        # Gradient
        W.append(self.w[0] - self.alpha * np.kron(delta[0], x).reshape(len(self.z[0]), x.shape[0]))
        B.append(self.b[0] - self.alpha * delta[0])
        
        for i in range(1, len(self.z)):
            W.append(self.w[i] - self.alpha * np.kron(delta[i], self.z[i-1]).reshape(len(self.z[i]), len(self.z[i-1])))
            B.append(self.b[i] - self.alpha * delta[i])
        

        return W, B

    def reset(self):
        self.w = []
        self.b = []
        self.act_f = []

    def fit(self, X, Y, verbose=False, epochs=1, 
            normalization={}, _print=False, 
            X_test=None, Y_test=None, 
            early_stopping=None, 
            precision=5):

        self.X = X
        self.Y = Y

        self.X_test = X_test
        if self.X_test is None:
            self.X_test = self.X
        self.Y_test = Y_test
        if self.Y_test is None:
            self.Y_test = self.Y

        self.normalization = normalization
        self.epochs = epochs

        # Early stopping
        it = 0
        min_loss = np.inf


        if len(self.layers) < 2:
            raise RuntimeError("Error, mlp needs a minimum of an input and an output layer to perform")

        self.reset()

        self.w.append(np.random.randn(self.layers[0].size , self.X.shape[1]))
        self.b.append(np.random.randn(self.layers[0].size))
        self.act_f.append(self.layers[0].get_act_f())
        # print("Input layer shape: ", self.w[0].shape, self.b[0].shape)

        if _print is True:
            print(f'{self.layers[0].label}', 0)
            print('Number of neurons: ', self.layers[0].size)
        if verbose is True:
            print(self.w[0], self.b[0])

        for i in range(1, len(self.layers)):
            if _print is True:
                print(f'{self.layers[i].label}', i)
                print('Number of neurons: ', self.layers[i].size)
            self.w.append(np.random.randn(self.layers[i].size , self.layers[i-1].size))
            self.b.append(np.random.randn(self.layers[i].size))
            self.act_f.append(self.layers[i].get_act_f())
            # print("w shape : ", self.w[i].shape, " b shape: ", self.b[i].shape)
            if verbose is True:
                print(self.w[i], self.b[i])
        
        for epoch in range(0, epochs):
            for i in range(0, self.X.shape[0]):
                # print(self.X[i])
                '''
                Now we start the feed forward
                '''  
                self.z = []

                self.z.append(self.feed_forward(self.w[0], self.b[0], self.act_f[0], self.X[i]))
                # print(self.z[0])
                for j in range(1, len(self.layers)):
                    self.z.append(self.feed_forward(self.w[j] , self.b[j], self.act_f[j], self.z[j-1]))
                # print(self.z)
                self.w, self.b = self.back_propagation(self.X[i], self.Y[i])

            y_pred = self.predict(self.X_test, raw=True)
            y_pred_hs = np.heaviside(np.array(y_pred) - self.treshold, self.treshold)

            self.metrics['losses'].append(((self.Y_test - y_pred)**2).mean())
            self.metrics['scores'].append(accuracy_score(self.Y_test, y_pred_hs))
            self.metrics['binary_cross_entropy'].append(self.binary_cross_entropy(y_pred, self.Y_test))

            # self.scores.append(accuracy_score(self.Y, y_pred_hs))
            if _print is True:
                print(f"{epoch+1}/{epochs}:", end=" - ")
                print(f"r2: {r2_score(self.Y_test, y_pred)}", end=" - ")
                print(f"loss: {self.metrics['losses'][-1]}", end=" - ")
                print(f"score: {self.metrics['scores'][-1]}", end=" - ")
                print(f"Log loss: {self.metrics['binary_cross_entropy'][-1]}")

            if early_stopping is not None:
                loss = self.metrics['losses'][-1]
                if np.round(loss, precision) < np.round(min_loss,precision):
                    min_loss = loss
                    count = 0
                else:
                    count += 1
                    if count >= early_stopping:
                        print('early stopping at iteration : ', epoch+1)
                        break
        self.epochs = len(self.metrics['losses'])
        
    def predict(self, X, raw=False):
        y_pred = []
        for i in range(0, X.shape[0]):
            z = []
            z.append(self.feed_forward(self.w[0], self.b[0], self.act_f[0], X[i]))
            for i in range(1, len(self.layers)):
                z.append(self.feed_forward(self.w[i] , self.b[i], self.act_f[i], z[i-1]))
            if z[-1].shape[0] == 1:
                y_pred.append(z[-1][0])
            else:
                y_pred.append(z[-1])
        if raw == False:
            y_pred = np.heaviside(np.array(y_pred) - self.treshold, self.treshold)
        return np.array(y_pred)

    def describe(self):
        pass
    
    def binary_cross_entropy(self, p, y, e=1e-15):
        r = 0
        for i in range(0, len(p)):
            r += (np.dot(y[i], np.log(p[i]+e))) + (np.dot((1 - y[i]), np.log(1 - p[i]+e)))
        r = -r / len(p)
        return r

    def score():
        pass

    def export(self, path="./mlp.pkl"):
        pickle.dump(self, open(path, "wb+"))
        print(f"successfuly exported in {path}")