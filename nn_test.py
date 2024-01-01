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


# Other activation function other than relu
    # Sigmoid Function
    ## Sigma(x) =1/1+e^x

# Application: Binary classification (e.g., spam detection).

# Why: It squashes the input into a range between 0 and 1, making it interpretable as a probability.
# How: Often used in the final layer of a binary classification network.
    
# Example input (logit)
logit = torch.tensor([0.5])
# Applying Sigmoid
prob = F.sigmoid(logit)
print(prob)  # Interpreted as probability





    # Tanh Function
    ## tanh(x)= e^e-e^-x/e^x+e^-x

# Application: Intermediate layers in a network, especially in RNNs.


# Why: Outputs range between -1 and 1, centering the data which can be beneficial in certain architectures.
# How: Often used in RNNs to regulate the flow of information.

# Example input
input_tensor = torch.tensor([0.5])
# Applying Tanh
tanh_output = F.tanh(input_tensor)
print(tanh_output)





    # Leaky Relu Function
    ## LReLU(x) = max(0.01x,x)

# Application: Addressing dying ReLU problem in deep networks.

# Why: Allows a small, non-zero gradient when the unit is not active.
# How: Used as an alternative to ReLU, especially in deeper networks.

# Example input
input_tensor = torch.tensor([-0.5])
# Applying Leaky ReLU
leaky_relu_output = F.leaky_relu(input_tensor, negative_slope=0.01)
print(leaky_relu_output)



# Mooving to loss functions
    # Mean Square error
    ## MSE(y, y') = 1/n SUM(y-y')^2

# Application: Regression tasks (e.g., predicting house prices).

# Why: Measures the average of the squares of the errors between actual and predicted values.
# How: Used as a loss function in regression models.

# Target and predicted values
target = torch.tensor([2.5])
prediction = torch.tensor([2.0])
# Applying MSE Loss
mse_loss = F.mse_loss(prediction, target)
print(mse_loss)



    # Cross Entropy Function
    ## -(y*log(p) + (1-y)*log(1-p))

# Application: Multi-class classification (e.g., image classification).

# Why: Measures the performance of a classification model whose output is a probability.
# How: Commonly used in the final layer of classification networks.

# Example targets and predictions
target = torch.tensor([1, 0, 1])
predictions = torch.tensor([[0.25, 0.75], [0.60, 0.40], [0.30, 0.70]])
# Applying Cross-Entropy Loss
cross_entropy_loss = F.cross_entropy(predictions, target)
print(cross_entropy_loss)



# Moving to pooling functions
    # Max Pooling 2D
    # Formula: Selects the maximum value from a portion of the input tensor.
    # Application: Convolutional Neural Networks for image processing.

# Why: Reduces the spatial dimensions (width, height) of the input volume, making the computation more efficient.
# How: Commonly used after convolutional layers in CNNs.

# Example 2D tensor (image)
input_tensor = torch.tensor([[[[4, 3], [2, 1]]]])
# Applying Max Pooling
max_pool_output = F.max_pool2d(input_tensor, kernel_size=2)
print(max_pool_output)


    # Average pooling 2D
    # Formula: Computes the average of a portion of the input tensor.
    # Application: Convolutional Neural Networks, often used for feature downsampling.

# Why: Similar to max pooling, but computes the average, which can be less aggressive.
# How: Used in CNNs as an alternative to max pooling.

# Example 2D tensor
input_tensor = torch.tensor([[[[4, 3], [2, 1]]]])
# Applying Average Pooling
avg_pool_output = F.avg_pool2d(input_tensor, kernel_size=2)
print(avg_pool_output)


    # Batch Normalization
    # Formula: Normalizes the output of the previous layer by subtracting the batch mean and dividing by the batch standard deviation.
    # Application: Used in various network architectures to improve training speed and stability.

# Why: Helps in reducing internal covariate shift.
# How: Applied to the output of a layer, usually before the activation function.

# Example 2D tensor
input_tensor = torch.randn(1, 2, 2, 2)  # Random tensor
# Applying Batch Normalization
batch_norm_output = F.batch_norm(input_tensor, running_mean=None, running_var=None)
print(batch_norm_output)


    # Dropout
    # Formula: Randomly zeroes some of the elements of the input tensor with probability p during training.
    # Application: Used as a regularization technique to prevent overfitting.

# Why: Prevents the model from being too dependent on any one feature.
# How: Applied to the output of a layer, usually after the activation function.

# Example input
input_tensor = torch.randn(1, 10)  # Random tensor
# Applying Dropout
dropout_output = F.dropout(input_tensor, p=0.2)
print(dropout_output)

