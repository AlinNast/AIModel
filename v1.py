import torch

x = torch.rand(2, 4, 7, 4)
y = torch.rand(6,512,512,4)
a = torch.tensor([3,1.5,1.5,0.5])

print(x*a)
print(y*a)