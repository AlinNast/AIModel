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
    
def generateName():
    out = []
    ix = 0 # so it start on the row of N that represents chars that follow . (aka first letters)
    
    while True:
        p = N[ix].float()
        p = p/p.sum()   # this normalises the values of p, aka makes every value in the array the probality of it to appear
        
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
    
    print("\nThe first row of N")
    print(N[0])
    
    print("\nWith the N representing the map of probablilities of each letter following eachother we can now generate a name:")
    
    for i in range(5):
        print(''.join(generateName()))
    
    

if __name__ == "__main__":
    main()