import random
from typing import Any
from engine import Value



class Neuron():
    """ This represents a neuron, it stores a array of Values (w), a bias and produces  (w*x+b).tanh"""
    
    def __init__(self, nin, nonlin=True):
        # nin ~ Number of INputs
        
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)] 
        # stores the number of weights in a array from the inputs it gets
        
        self.b = Value(0) # Bias - Sets The trigger happines of the neuron
        
    def __call__(self, x):
        """ calculates (and returns) w * x + b for every weight. size of x.lenght should match nin"""
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        
        out = act.relu()
        
        return out
    
    def parameters(self):
        """This function of a neuron returns its array of weight and its bias at the end"""
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Linear Neuron({len(self.w)})"
    
class Layer():
    """ This is a layer of neurons, it stores a array, upon call it performs the neurons math"""

    def __init__(self, nin, nout):
        # Initialisez a layer of neurons(with the nin) of size equal to the number of outputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """it returns the value of x computed by the array of neurons in a array"""
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        """This function of a Layer returns a list of the containing neurons weights"""
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP():
    """ This is a array of layers (MultyLevelPerceptron)"""
    
    def __init__(self, nin, nouts):
        # this instantietes the layers of neruons
        sz = [nin] + nouts # = [nin, nout[0]...]
        # Creates a nr of layers = to nouts
        # Listifies the nr of layers we want in the mlp
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # calls the layer with the value x sequencially
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """This returns a array of the layers containing the neurons containing the weights"""
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
# n = MLP(3, [4, 4, 1])
# this is a peceptron with a 3 dimension neuron input that gows to a 
# layer of 4 neuron, another layer of 4 neurons, and then a layer of 1 neuron 
# which is the output

# Lets try to run the program, initialize MLP, and play with test data, desired data and predicted data
#
# call in the mlp on the training test data, get the predicted data,compare it to the desider data
# 
# create the loss function, in this case mean square error, get the loss value
# Call the loss.backward() and get every gradient of every value in the neurons in the layers 
#

# notes for the test: 
# instantiete n
# create array to match input size (training data)
# create desired output
# iterate: call array with training data, get predicted data, calculate loss function (MSE),
# zero grads, calculate loss backward,  interate through every params and ajust data with (new) gradient,
# redo iteration
# 
# This has 41 parameters and a small array as a training data, but conceptually its AI just as much as a chat gpt
#that hasi billion(s) of parameters and training data as big as the internet