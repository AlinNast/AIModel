import os
import torch

names = open('names.txt', 'r').read().splitlines() # This saved the names from the dataset in a list
# names = open('.\GitHub\AIModel\BigramLm\\names.txt', 'r').read().splitlines()

N = torch.zeros((28,28), dtype=torch.int32) # this is the tensor that stores the characters, 28 becouse the alphabet hast 26 + our 2 special characters

chars = sorted(list(set(''.join(names))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
# stoi is a dictionary that stores every character + the 2 special ones with a coresponding index in alphabetical order


def main():
    print("Program Initialized")
    
    for name in names:
        characters = ['<S>'] + list(name) + ['<E>'] # S marks the start of a name and E marks the end
        for ch1 ,ch2 in zip(characters, characters[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1,ix2] += 1
            # They way this works, it has a list(row) for every character, and that row has a list (column) for every combination with another character
            # this way every value in the tensor represents how many times thay characters index in the column met with the characters index the column has in the big row 
    
    print(N)
    

if __name__ == "__main__":
    main()