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

block_size = 3


def build_dataset(names):
    block_size = 3 # context lenght - how many characters are taken into consideration for the next prediction
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
def split_loss(C, W1, b1, W2, b2,batch_normalization_gain, batch_normalization_bias):
    emb = C[Xdev]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = batch_normalization_gain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + batch_normalization_bias

    h = torch.tanh(hpreact) # N to H1 activation
    logits = h @ W2 + b2 # H1 to Out Activation
    loss = F.cross_entropy(logits, Ydev)
    print(f"Evaluation loss: {loss}")
    
def generate_name(C, W1, b1, W2, b2,batch_normalization_gain, batch_normalization_bias):
    out = []
    context = [0] * block_size
    
    while True:
        emb = C[torch.tensor([context])]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1 + b1
        
        h = torch.tanh(batch_normalization_gain * hpreact+ batch_normalization_bias) # N to H1 activation
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
    
random.shuffle(names)
lim1 = int(0.8*len(names))
lim2 = int(0.9*len(names))
Xtr, Ytr = build_dataset(names[:lim1])
Xdev, Ydev = build_dataset(names[lim1:lim2])
Xtest, Xtest = build_dataset(names[lim2:])


def main():
    ### Main characteristics of the MLP
    n_embed = 10   # The dimensionaliti of the character embeding vectors
    n_hidden = 200   # The number of neurons in the hidden layer of the MLP
    
    training_steps = 20001
    batch_size = 32   # The size of the minibatch training sets
    
    
    tic = time.perf_counter()
    print("MLP Initialized")


    ### Layers of the MLP
    C = torch.randn(vocab_size, n_embed)   # embeding layer
    W1= torch.randn(n_embed * block_size, n_hidden) * (5/3)/((n_embed*block_size)**0.5) # Hidden layer
    b1= torch.randn(n_hidden) * 0.01  # Hidden layer Biases
    W2= torch.randn(n_hidden, vocab_size) * 0.01  # output layer
    b2= torch.randn(vocab_size) * 0  # output Layer Biases
    ### Batch Normalization parameters
    batch_normalization_gain = torch.ones((1, n_hidden)) # used in the initialization to normalize the preactivation in a gaussian distribution
    batch_normalization_bias = torch.zeros((1, n_hidden)) # same
    bn_mean_running = torch.zeros((1, n_hidden))
    bn_std_running = torch.ones((1, n_hidden))
    
    print(f"Layers created in {time.perf_counter() - tic} sec")
    
    parameters = [C, W1, b1, W2, b2, batch_normalization_gain, batch_normalization_bias]
    print(f"This MLP contain {sum(p.nelement() for p in parameters)} parameters")
    for p in parameters:
        p.requires_grad=True
        
    
    ### Training of the MLP
    for i in range(training_steps):
        
        # Construct minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,)) # this creates a tensor with randoms between 0 and 32 (based on X) of size 32 to act as indexes for the batch
        Xbatch, Ybatch = Xtr[ix], Ytr[ix]  # Batch resulted from indexig the trainig data
        
        # Forward pass
        emb = C[Xbatch] # Embeded characters into vecotors
        embcat = emb.view(emb.shape[0], -1) # Concatenate Vectors
        # Linear Layer
        h_preact = embcat @ W1 +b1 # Hidded layer pre activation
        # Batch Normalization Layer
        bn_mean_i = h_preact.mean(0, keepdim=True)
        bn_std_i = h_preact.std(0, keepdim=True)
        h_preact = batch_normalization_gain * (h_preact - bn_mean_i) / bn_std_i + batch_normalization_bias
        with torch.no_grad():
            bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_mean_i
            bn_std_running = 0.999 * bn_std_running + 0.001 * bn_std_i
        # NonLinearity
        h = torch.tanh(h_preact) # Hidden Layer
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ybatch)
        
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # Update
        lr = 0.06 if i < training_steps/2 else 0.0003
        for p in parameters:
            p.data += -lr * p.grad
        
        # Info
        if i % 10000 == 0:
            print(f"Value of the loss funtion: {loss.item()} on training step:{i}")
    
    print(f"Training of {training_steps} steps completed {time.perf_counter() - tic} sec")   
    
    # Evaluate training
    split_loss(C, W1, b1, W2, b2,batch_normalization_gain,batch_normalization_bias)
    
    # Generate name
    generate_name(C, W1, b1, W2, b2,batch_normalization_gain,batch_normalization_bias)
    
            


if __name__ == "__main__":
    main()