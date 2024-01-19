import os
import torch
import time
import torch.nn.functional as F

#names = open('names.txt', 'r').read().splitlines() # This saved the names from the dataset in a list
names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32) # this is the tensor that stores the characters, 28 becouse the alphabet hast 26 + our 2 special characters


chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
# stoi is a dictionary that stores every character + the 2 special ones with a coresponding index in alphabetical order

def mapN():
    maxMap = (len(names))
    for name in names[:maxMap]:
        characters = ['.'] + list(name) + ['.'] # S marks the start of a name and E marks the end
        for ch1 ,ch2 in zip(characters, characters[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1,ix2] += 1
            # They way this works, it has a list(row) for every character, and that row has a list (column) for every combination with another character
            # this way every value in the tensor represents how many times thay characters index in the column met with the characters index the column has in the big row 
            
def evaluateQuality(name):
    log_likelihood = 0.0
    count = 0
    
    P = (N + 1).float()  # the N+1 is a technique called model smoothing, it makes it so that there are no 0s in the matrix, this way no -infinity could appear in the loss function
    P /= P.sum(1,keepdim=True) # this normalises the values of p, aka makes every value in the array the probality of it to appear
  
    for ch1 ,ch2 in zip(name, name[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        count +=1
        
    nll = -log_likelihood # this is a way of definig a loss function for a bigram
    # negative log likelihood is: the sum of the normalized(by log) probabilitiesmultiplied by -1 (so it will be positive)
    # closer to 0 means better prediction
    return nll
    
def generateName():
    out = []
    ix = 0 # so it start on the row of N that represents chars that follow . (aka first letters)
    
    P = (N + 1).float()  # the N+1 is a technique called model smoothing, it makes it so that there are no 0s in the matrix, this way no -infinity could appear in the loss function
    P /= P.sum(1,keepdim=True) # this normalises the values of p, aka makes every value in the array the probality of it to appear
    
    while True:
        p = P[ix]   
        
        ix = torch.multinomial(p, num_samples=1, replacement=True).item() # this gets a index from the array according to its probability aka the index of the generetad letter
        out.append(itos[ix])
        
        if ix == 0:
            break
    
    return out

def generateAIname(W):
    
    out = []
    ix = 0
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]),num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims = True)
        
        ix = torch.multinomial(p, num_samples=1, replacement=True).item() # this gets a index from the array according to its probability aka the index of the generetad letter
        out.append(itos[ix])
        
        if ix == 0:
            break
    return out

def create_train_dataset():
    ### Similar to mapping N, in order to prepare the data for a neural net we need a array of integers to ilustrate what characters follow another, 
    # rather than one array to count how many times it gets followed 
    
    print(f"Dataset contains {len(names)} names for training a bigram model")
    xs = []
    ys = []
    for name in names:
        characters = ['.'] + list(name) + ['.'] # S marks the start of a name and E marks the end
        for ch1 ,ch2 in zip(characters, characters[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys
    
def encoder():
    xs, ys = create_train_dataset()
    xenc = F.one_hot(xs,num_classes=27).float()
    return xenc, ys


    

def main():
    print("Program Initialized")
    
    # print("\nStart dataset mapping")
    # tic= time.perf_counter()
    # mapN()
    # tac = time.perf_counter()
    # print(f"Dataset mapped in {tac - tic} seconds")
    # print("\n Name generation based on probability")
    
    print("\nENCODE the data")
    
    tic= time.perf_counter()
    xenc, ys = encoder()
    tac = time.perf_counter()
    print(f"Dataset encoded in {tac - tic} seconds")

    print("\n Creating Neuron Layer ")
    # This is THE Neuron
    W = torch.randn((27,27), requires_grad=True)    # This creates a tensor with random values froma normalized function
    
    print("\n Ganerating test names before model training")
    for i in range(5):
        name = generateAIname(W=W)
        print(''.join(name))
        
        
    print("\n \n Initiallizing gradient descent")
    tic= time.perf_counter()
    for k in range(321):
        
        ### basicly the forward pass
        logits =  xenc @ W  # this provides a tensor that represent the log counts of the characters on the training data
        #probs = softmax(logits=logits)
        counts = logits.exp() # this exponentietes the vectors (0-1 for negative vectors, 1+for positive) thus preparing the data to be transformed into probabilities
        # this is equivalent to N
        Probs = counts / counts.sum(1,keepdim=True)  # This transforms normalized counts into probabilieties, 
        # this is equivalent to P
       
        loss= -Probs[torch.arange(len(xenc)), ys].log().mean()
        
        if k % 80 == 0:
            print(f"Epoch: {k}, loss: {loss.data}")
        
    
        ### basicly the backward pass
        W.grad = None
        loss.backward()
    
        ### Basicly the update
        W.data += -8 * W.grad
    
    tac = time.perf_counter()
    print(f"Neural network trained in {tac - tic} seconds")
    
    print("\n Ganerating names after model training")
    for i in range(5):
        name = generateAIname(W=W)
        print(''.join(name))
    

if __name__ == "__main__":
    main()