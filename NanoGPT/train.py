import torch
import torch.nn as nn
from torch.nn import functional as F


## This opens the training data. 
with open('.\GitHub\AIModel\\NanoGPT\\train.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    

# get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Tokenizing the encoded data into a tensor and splitting it into train and evaluation sets
data = torch.tensor(encode(text), dtype=torch.long)
data_split = int(0.9*len(data))
train_data = data[:data_split]
val_data = data[data_split:]

batch_size = 4 # how many independent sequences will be processed in paralel (4 becouse my cpu is quad core)
block_size = 8 # The lenght of the context for predictions

def get_batch(split):
    # Generate a small batch of data of Inputs X and target y. 
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Basically random Offsets into the data set With a dimension of 4  ( batch_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')


# super simple bigram model, already built by torch.nn module
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)

