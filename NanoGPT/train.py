


## This opens the training data. 
with open('.\GitHub\AIModel\\NanoGPT\\train.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    

# get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")