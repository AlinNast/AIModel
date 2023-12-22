import torch


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

# Reshaping A
C = A.view(1, 3)  # C is now a 1x3 tensor
print("Dimentionality \n")
print(C)



#In place operations
A = torch.tensor([1, 2, 3])

# In-place Addition
A.add_(5)  # A is now [6, 7, 8]
