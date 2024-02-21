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

block_size = 3  # context lenght - how many characters are taken into consideration for the next prediction


# Build Dataset function
def build_dataset(names):
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


random.shuffle(names)
lim1 = int(0.8*len(names))
lim2 = int(0.9*len(names))
Xtr, Ytr = build_dataset(names[:lim1])  # 80% dataset
Xdev, Ydev = build_dataset(names[lim1:lim2])  # 10% dataset
Xtest, Xtest = build_dataset(names[lim2:])  # 10% dataset

# A function used to calculate the diff between gradients
def cmp(s, dt, t):
    ex = torch.all(dt==t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt-t.grad).abs().max().item()
    print(f"exact match:{ex}, approximate match:{app}, maxdiff:{maxdiff}\n")

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
    



def main():
    ### Main characteristics of the MLP
    n_embed = 10   # The dimensionaliti of the character embeding vectors
    n_hidden = 50   # The number of neurons in the hidden layer of the MLP
    
    training_steps = 1
    batch_size = 32   # The size of the minibatch training sets
    
    
    tic = time.perf_counter()
    print("MLP Initialized")


    ### Layers of the MLP
    # embeding layer
    C = torch.randn(vocab_size, n_embed)   
    # Hidden layer
    W1= torch.randn(n_embed * block_size, n_hidden) * (5/3)/((n_embed*block_size)**0.5) 
    b1= torch.randn(n_hidden) * 0.1  # Hidden layer Biases
    # Output layer
    W2= torch.randn(n_hidden, vocab_size) * 0.1  
    b2= torch.randn(vocab_size) * 0.1  # output Layer Biases
    ### Batch Normalization parameters
    batch_normalization_gain = torch.ones((1, n_hidden)) # used in the initialization to normalize the preactivation in a gaussian distribution
    batch_normalization_bias = torch.zeros((1, n_hidden)) # same
    
 
    
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
        
        # Forward pass ELABORATED
        emb = C[Xbatch] # Embeded characters into vecotors
        embcat = emb.view(emb.shape[0], -1) # Concatenate Vectors
        
        # Linear Layer 1 
        h_preact_bn = embcat @ W1 +b1 # Hidded layer pre activation
        # Batch Normalization Layer 1
        bn_mean_i = 1/batch_size*h_preact_bn.sum(0, keepdim=True)
        bn_diff = h_preact_bn-bn_mean_i
        bn_diff2 = bn_diff**2
        bn_var = 1/(batch_size-1)*(bn_diff2).sum(0, keepdim=True)
        bn_var_inv = (bn_var + 1e-5)**0.5
        bn_raw = bn_diff * bn_var_inv
        h_preact = batch_normalization_gain * bn_raw +batch_normalization_bias
        
        
        # NonLinearity
        h = torch.tanh(h_preact)  # Hidden Layer
        
        # Layer 2
        logits = h @ W2 + b2  # Output Layer
        
        # Calculate Loss functions
        # loss = F.cross_entropy(logits, Ybatch)
        logit_maxes = logits.max(1, keepdim=True).values
        norm_logits = logits - logit_maxes # dlogit_maxes = -dnorm_logits.sum(1)
        counts = norm_logits.exp() # dnorm_logits = counts
        counts_sum = counts.sum(1, keepdim=True)  # idk
        counts_sum_inv = counts_sum**-1  # dcounts_sum = (-counts_sum**-2)
        probs = counts *counts_sum_inv  # dcounts_sum_inv = counts.sum(1)
        log_probs = probs.log()  # dprobs = 1/probs
        loss = -log_probs[range(batch_size), Ybatch].mean()  # dlog_probs[range(batch_size, Ybatch)] = -1/batch_size 
        
        # Backward pass
        for p in parameters:
            p.grad = None
            
        for t in [log_probs, probs, counts,counts_sum, counts_sum_inv, norm_logits, logit_maxes,logits, h,h_preact, bn_var, bn_diff2, bn_diff,h_preact_bn, bn_mean_i,embcat,emb]:
            t.retain_grad()
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
    #split_loss(C, W1, b1, W2, b2,batch_normalization_gain,batch_normalization_bias)
    
    # Generate name
    #generate_name(C, W1, b1, W2, b2,batch_normalization_gain,batch_normalization_bias)
    
    print(f"Calculate log_probs derivative and compare it with the one calculated by loss.backwards")
    dlog_probs = torch.zeros_like(log_probs)
    dlog_probs[range(batch_size), Ybatch] = -1.0/batch_size
    cmp('log_probs',dlog_probs, log_probs)    
    
    print(f" Calculating probs derivative chained with the derivative of dlog_probs | 1/probs * dlog_probs ")
    dprobs = (1.0/probs) * dlog_probs
    cmp('probs',dprobs, probs)
    
    print(f" Calculating counts_sum_inv derivative ")
    dcounts_sum_inv = (counts*dprobs).sum(1, keepdim=True)
    cmp('dcounts_sum_inv',dcounts_sum_inv, counts_sum_inv)
    
    print(f" Calculating counts_sum derivative ")
    dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
    cmp('dcounts_sum',dcounts_sum, counts_sum)
    
    print(f" Calculating counts derivative ")
    dcounts = counts_sum_inv *dprobs
    dcounts += torch.ones_like(counts) * dcounts_sum
    cmp('dcounts',dcounts, counts)
    
    print(f" Calculating norm_logits derivative ")
    dnorm_logits = counts * dcounts
    cmp('dnorm_logits',dnorm_logits, norm_logits)
    
    print(f" Calculating logit_maxes derivative ")
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    cmp('dlogit_maxes',dlogit_maxes, logit_maxes)
    
    print(f" Calculating logit derivative ")
    dlogits = dnorm_logits.clone()
    dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
    cmp('dlogits',dlogits, logits)
    
    
    
    
            


if __name__ == "__main__":
    main()