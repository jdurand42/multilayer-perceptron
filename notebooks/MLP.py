import numpy as np

# Tuto
# simple-nn-with-python-multi-layer-perceptron

def _sigmoid(x, der=False):
    # perform sigmoid
    if der is False:
        r = 1 / (1 + np.exp(-x))
    else:
        r = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    return r

class MultiLayerPerceptron:
    class Layer:
        def __init__(self, size=1, act_f="sigmoid", label="input_layer"):
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
            else:
                raise TypeError(f"Error: unknown activation function: {self.f}")
        
        def size():
            return self.size

    def __init__(self, seed=10, alpha=0.1, layers=[]):
        self.w = []
        self.b = []
        self.layers = []
        self.mu = []
        self.seed = seed
        self.act_f = []
        self.alpha = alpha

        np.random.seed(self.seed)

        if len(layers) > 0:
            for i in range(0, len(layers)):
                self.add_layer(size=layers[i][0], act_f=layers[i][1], label=layers[i][2])

    # Activations:

    def add_layer(self, size=1, act_f="sigmoid", label="input_layer"):
        self.layers.append(self.Layer(size=size, act_f=act_f, label=label))
    
    def feed_forward(self, w, b, act_f, x):
        return act_f(np.dot(w, x) + b)

    def fit(self, X, Y, verbose=False, epochs=1):
        # fit
        # 
        # init of w and b for each layer
        print("init of layers")

        print("init of input layer")

        self.X = X
        self.Y = Y

        if len(self.layers) < 2:
            raise RuntimeError("Error, mlp needs a minimum of an input and an output layer to perform")

        self.w.append(np.random.randn(self.layers[0].size , self.X.shape[1]))
        self.b.append(np.random.randn(self.layers[0].size))
        self.act_f.append(self.layers[0].get_act_f())
        print("Input layer shape: ", self.w[0].shape, self.b[0].shape)
        if verbose is True:
            print(self.w[0], self.b[0])

        for i in range(1, len(self.layers)):
            print('Hidden Layer ', i)
            print('Number of neurons: ', self.layers[i].size)
            print(self.layers[i].size, self.layers[i].size)
            self.w.append( np.random.randn(self.layers[i].size , self.layers[i-1].size ))
            self.b.append( np.random.randn(self.layers[i].size))
            self.act_f.append(self.layers[i].get_act_f())
            print("w shape : ", self.w[i].shape, " b shape: ", self.b[i].shape)
            if verbose is True:
                print(self.w[i], self.b[i])


        # print("init Output layer ", len(self.layers) - 1)
        # print('Number of neurons: ', self.layers[-1].size)
        # self.w.append(np.random.randn(self.layers[-1].size , self.layers[1].size )/np.sqrt(2/self.layers[-2].size))
        # self.b.append(np.random.randn(self.layers[-1].size)/np.sqrt(2/self.layers[-2].size))
        # print("w shape : ", self.w[-1].shape, " b shape: ", self.b[-1].shape)
        # self.act_f.append(self.layers[-1].get_act_f())
        # if verbose is True:
            # print(self.w[-1], self.b[-1])
        
        for i in range(0, epochs):
            for i in range(0, 5):
                # print(self.X[i])
                '''
                Now we start the feed forward
                '''  
                self.z = []

                self.z.append(self.feed_forward(self.w[0], self.b[0], self.act_f[0], self.X[i])) # First layers
                # print(self.z[0])
                for i in range(1, len(self.layers)): #Looping over layers
                    self.z.append(self.feed_forward(self.w[i] , self.b[i], self.act_f[i], self.z[i-1]))
                print(self.z)


        # feed foward

        # backprop

        pass

    def predict(self, X):
        # predict 
        pass

    def describe(self):
        pass
