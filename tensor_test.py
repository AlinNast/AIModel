import torch


### Operation rules

# Element wise operations
A = torch.tensor([1, 2, 3])
B = torch.tensor([4, 5, 6])

# Element-wise Addition
C = A + B  # C will be [5, 7, 9]

# Element-wise Multiplication
D = A * B  # D will be [4, 10, 18]


# Broadcasting
A = torch.tensor([1, 2, 3])
B = 2

# Broadcasting B over A for Multiplication
C = A * B  # C will be [2, 4, 6]


#Matix Multiplication
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Matrix Multiplication
C = torch.matmul(A, B)  # or C = A @ B
print("matmul operation \n")
print(C)


#Dimensionality
A = torch.tensor([1, 2, 3])

# Adding a scalar (dimensionality changes)
B = A + 2  # B will be [3, 4, 5]


A = torch.tensor([1, 2, 3])
# Reshaping A
C = A.view(1, 3)  # C is now a 1x3 tensor
print("Dimentionality \n")
print(C)



#In place operations
A = torch.tensor([1, 2, 3])

# In-place Addition
A.add_(5)  # A is now [6, 7, 8]



### Common Operations

# Reshaping
A = torch.arange(1, 10)  # A tensor: [1, 2, 3, 4, 5, 6, 7, 8, 9]
B = A.view(3, 3)  # Reshape to a 3x3 matrix
print("Reshaping a 1 dimension to 3x3\n")
print(B)

# In a CNN, you might reshape a flat image tensor to a 3D tensor (channels, height, width).

# Slicing
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("slicing is a easy condept, learned during exercises on codewars \n")
# Selecting the first row
print(A[0])

# Selecting a specific element
print(A[1, 1])  # 5


# Aggregation
A = torch.tensor([[1, 2], [3, 4]])

# Compute the mean
print("aggregation \n")
print("mean")
#mean_val = A.mean()
#print(mean_val)
print("sum")
print(A.sum())
print("max")
print(A.max())

# Often used in calculating loss functions or metrics


# Element wise math functions
A = torch.tensor([1, 2, 3])

# Applying exponential function
print("Applying exponential function")
print(torch.exp(A))

# You might see this in normalizing data or in certain types of layers in neural networks.


# Normalization
A = torch.tensor([1.0, 2.0, 3.0])

# Subtract mean and divide by standard deviation
normalized_A = (A - A.mean()) / A.std()
print("Normalization( Subtract mean and divide by standard deviation ) \n")
print(normalized_A)


# Crucial in preparing data for neural networks, ensuring consistent scale across features.
