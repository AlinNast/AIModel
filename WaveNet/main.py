import torch
import torch.nn.functional as F
import matplotlib
import random
import sys
import time


names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()

# Build the vocabulary of characters and map from string to int and vice versa
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)} # string to int dict
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # int to string dict
vocab_size = len(itos)


def build_dataset(names, block_size=3):
    block_size = block_size # context lenght - how many characters are taken into consideration for the next prediction
    X = []   # the X is the input of the nn
    Y = []   # The Y is the laver for each example in X
    for n in names:
        #print(n)
        context = [0] * block_size  # makes . if there is no prev context
        for ch in n + '.':
            ix = stoi[ch]
            X.append(context)  # this stores the context of each character as a int
            Y.append(ix)       # this stores the index of the character to whom the context is
            context = context[1:] + [ix]  # updates the context to the next char
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X , Y

@torch.no_grad()
def split_loss(x,y,model):
    x,y = x,y
    logits = model(x)
    loss = F.cross_entropy(logits,y)
    return loss

def generate_name(block_size=3, model=None):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))


class Linear:
    """Building a Linear layer of neurons, performs x@W+b"""
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # kaimin init
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight # forward pass x @ W
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class BatchNorm1d:
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # params (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers ( trained with momentum)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class Tanh:
    
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []


class Embedding:
    
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings,embedding_dim))
        
    def __call__(self,IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    

class Flatten:
    
    def __init__(self,n):
        self.n =n
    
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
        
        
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x) -> torch.Any:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def main():
    print("Program Initialized: WaveNet")
    tic = time.perf_counter()
    
    # Building the dataset for trainingand testing
    block_size = 8
    random.shuffle(names)
    lim1 = int(0.8*len(names))
    lim2 = int(0.9*len(names))
    Xtr, Ytr = build_dataset(names[:lim1], block_size=block_size)
    Xdev, Ydev = build_dataset(names[lim1:lim2], block_size=block_size)
    Xtest, Ytest = build_dataset(names[lim2:], block_size=block_size)
    
    
    print(f"\nStarting ANN contruction")
    
    n_emb = 10      #The dimenstion of the character embeding layer
    n_hidden = 200  # The dimantion of the hidden layer
    
    # C = torch.randn((vocab_size, n_emb)) # embeding layer replaces by embeding module
    model = Sequential([
                Embedding(vocab_size, n_emb),
                Flatten(),
                Linear(n_emb*block_size, n_hidden, bias=False), 
                BatchNorm1d(n_hidden), 
                Tanh(), 
                Linear(n_hidden, vocab_size)])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1 # make last layer less confident
    
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    
    print(f"ANN Initialized with {sum(p.nelement() for p in parameters)}, in {time.perf_counter() - tic} seconds")
    
    print("\n Initializing BackPropagation")
    max_steps = 1000
    batch_size = 32
    
    for i in range(max_steps):
        # Construct minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,)) # this creates a tensor with randoms between 0 and 32 (based on X) of size 32 to act as indexes for the batch
        Xbatch, Ybatch = Xtr[ix], Ytr[ix]  # Batch resulted from indexig the trainig data
        
        # Forward pass
        logits = model(Xbatch)
        loss = F.cross_entropy(logits, Ybatch)
        
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # Update
        lr = 0.1
        for p in parameters:
            p.data += -lr * p.grad
            
        # Track stats
        if i % (max_steps/10) == 0:
            print(f"On trainig step {i} loss was {loss.item()}")

    print(f"Gradient descent Complete in {time.perf_counter()-tic}")
    
    print(f"\n Evaluating the loss")
    for layer in model.layers:
        layer.training = False
        
    print(f"Loss on training data: {split_loss(Xtr,Ytr,model)}")
    print(f"Loss on dev data: {split_loss(Xdev,Ydev,model)}")
    print(f"Loss on test data: {split_loss(Xtest,Ytest,model)}")

    print(f"\nGenerating Names with the model")
    generate_name(block_size,model)

if __name__ == "__main__":
    main()