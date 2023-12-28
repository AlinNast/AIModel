import torch

# Create tensors.
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

print("\nThis is x: \n")
print(x.grad) # this is none, it is part of the tensor object but it gets a value only after the backward of z is calculated
print("\nThis is y: \n")
print(y.grad) # this is none, it is part of the tensor object but it gets a value only after the backward of z is calculated

# Perform a computation.
z = x * x * y
print("This is 2x*y \n")
print(z)

# Compute gradients.
z.backward()

# x_grad = 2 * x * y
print(x.grad)  # Output: tensor([12.])
# y_grad = x * x
print(y.grad)  # Output: tensor([4.])

## ajusting the weights with the grad
learning_rate = 0.01

# Update weights.
x.data = x.data - learning_rate * x.grad
y.data = y.data - learning_rate * y.grad

print(x.data) # this is now tensor([1.8800]), also makes x=tensor([1.8800] requires_grad=True)

z = x * x * y
print("This is 2x*y now\n")
print(z)

# After the update, the weights x and y are adjusted in the direction that reduces z.
