import torch
import torch.nn.functional as F


names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()


# Build the vocabulary of characters and map from string to int and vice versa
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)} # string to int dict
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # int to string dict


# Build the DataSet
block_zise = 3 # context lenght - how many characters are taken into consideration for the next prediction
X = []   # the X is the input of the nn
Y = []   # The Y is the laver for each example in X
for n in names[:5]:
    #print(n)
    context = [0] * block_zise  # makes . if there is no prev context
    for ch in n + '.':
        ix = stoi[ch]
        X.append(context)  # this stores the context of each character as a int
        Y.append(ix)       # this stores the index of the character to whom the context is
        context = context[1:] + [ix]  # updates the context to the next char

X = torch.tensor(X)  # size 32,3
Y = torch.tensor(Y)  # size 32


# Build the embeding
C = torch.randn((27,2)) # the lookup table


# building the Hidden Layer and biases
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
    



# Building the final Layer
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
    


def main():
    print("started")
    parameters = [C, W1 ,b1, W2, b2]
    print(f"Program contains {sum(p.nelement() for p in parameters)}")
    print("Embading the input tensor")
    emb = C[X]  # size 32,3,2 # The embading of all the X values
    
    print("Activating the first Layer")
    # The activation of the first hidden layer
    h = torch.tanh(emb.view(-1,6) @ W1 + b1)  # size 32, 100
    # emp.view is a function of pytorch that changes the shape of emp to allow matrix multiplication without using concatination, thus bein more efficient
    
    print("Calculating the probability distribution  of the Output layer")
    # Building the output gathering logic
    logits = h @ W2 + b2   #size 32, 27
    # activastion of the final layer
    counts = logits.exp()  # simulates counts bases on the activation of the final layer (logits)
    probs = counts/counts.sum(1, keepdim=True)  # Normalize counts into probabilities
    
    print("calculating the loss function")
    loss = -probs[torch.arange(32), Y].log().mean()
    print(loss)


    print("finished")


if __name__ == "__main__":
    main()
    