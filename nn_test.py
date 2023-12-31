import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)   # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)    # Third fully connected layer

    def forward(self, x):
        x = F.relu(self.fc1(x))        # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))        # Apply ReLU activation function after second layer

        #ReLU, or Rectified Linear Unit, is an activation function, one of the most commonly used in neural networks. 
        #The function is defined as:
        # Relu(x) = max(0,x)
        
        x = self.fc3(x)                # Output layer
        return x
