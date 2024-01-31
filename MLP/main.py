import torch
import torch.nn.functional as F
import matplotlib
import random


names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()


# Build the vocabulary of characters and map from string to int and vice versa
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)} # string to int dict
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # int to string dict


# Build the DataSet
def build_dataset(names):
    block_zise = 3 # context lenght - how many characters are taken into consideration for the next prediction
    X = []   # the X is the input of the nn
    Y = []   # The Y is the laver for each example in X
    for n in names:
        #print(n)
        context = [0] * block_zise  # makes . if there is no prev context
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
Xtr, Ytr = build_dataset(names[:lim1])  # size 32,3
Xdev, Ydev = build_dataset(names[lim1:lim2])
Xtest, Xtest = build_dataset(names[lim2:])

# Build the embeding
C = torch.randn((27,10)) # the lookup table


# building the Hidden Layer and biases
W1 = torch.randn((30, 300))
b1 = torch.randn(300)
    



# Building the final Layer
W2 = torch.randn((300, 27))
b2 = torch.randn(27)
    


def main():
    print("started")
    parameters = [C, W1 ,b1, W2, b2]
    print(f"Program contains {sum(p.nelement() for p in parameters)} parameters")
    for p in parameters:
        p.requires_grad = True
    
                # This experiment shows how to find a apropriate learning rate
                # learning_rate_exponent = torch.linspace(-3,0,2000)
                # learning_rate_value = 10**learning_rate_exponent
    for i in range(20000):
        ### Creating Minibatches
        ix = torch.randint(0, Xtr.shape[0], (32,)) # this creates a tensor with randoms between 0 and 32 (based on X) of size 32
        # this way allowind the ANN to train on random small bathces of the dataset
        
        ### Forward pass bagin
        #print("Embading the input tensor")
        emb = C[Xtr[ix]]  # size 32,3,2 # The embading of all the X values
        
        #print("Activating the first Layer")
        # The activation of the first hidden layer
        h = torch.tanh(emb.view(-1,30) @ W1 + b1)  # size 32, 100
        # emp.view is a function of pytorch that changes the shape of emp to allow matrix multiplication without using concatination, thus being more efficient
        
        #print("Calculating the probability distribution  of the Output layer")
        # Building the output gathering logic
        logits = h @ W2 + b2   #size 32, 27
        # activastion of the final layer
        
        
                # counts = logits.exp()  # simulates counts bases on the activation of the final layer (logits)
                # probs = counts/counts.sum(1, keepdim=True)  # Normalize counts into probabilities
                # loss = -probs[torch.arange(32), Y].log().mean()
                ### This was used so far to substitute the cross entropy function
                
        #print("calculating the loss function")
        loss = F.cross_entropy(logits, Ytr[ix])
        ### End forward pass
        ### Backward pass begin
        for p in parameters:
            p.grad = None
        loss.backward()
        ### End Backward pass
        ### Update
        #lr = learning_rate_value[i]
        for p in parameters:
            p.data += -0.05 * p.grad
            #p.data += -lr * p.grad
        
        #print(lr)
        #print(loss.item())
    print("Evaluation of loss on training data after training")    
    print(loss.item())
    
    print("Evaluation on Development data after training")
    emb = C[Xdev] 
    h = torch.tanh(emb.view(-1,30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(loss.item())

    print("finished")


if __name__ == "__main__":
    main()
    