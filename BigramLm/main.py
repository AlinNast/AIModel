import os
import torch
import time

#names = open('names.txt', 'r').read().splitlines() # This saved the names from the dataset in a list
names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32) # this is the tensor that stores the characters, 28 becouse the alphabet hast 26 + our 2 special characters

chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
# stoi is a dictionary that stores every character + the 2 special ones with a coresponding index in alphabetical order

def mapN():
    for name in names:
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
        

def main():
    print("Program Initialized")
    
    print("\nStart dataset mapping")
    tic= time.perf_counter()
    mapN()
    tac = time.perf_counter()
    print(f"Dataset mapped in {tac - tic} seconds")
        
    while True:
        print("input pelase")
        i = input()
        
        name = generateName()
        if i == '1':
            print(''.join(name))
            print(f'The quality of the name = {evaluateQuality(name)}')
        elif i == 'e':
            break
        else:
            continue
    
    
    
    

if __name__ == "__main__":
    main()