from nn import MLP
import torch

# sample data
xs = [[10.0, 7.0, 9.0],[1.0, -10.0, -2] ]

#
ys = [1.0, 1.0]



def main():
    print("\nProgram Initialized\n")
    
    model = MLP(3 ,[4,4,1])
    print (model)
    
    print("\n   Now to call with test data:")
    print(xs)
    print("     And desired output:")
    print(ys)
    
    print("\n   The returned outputs:")
    ypred = [model(x) for x in xs]
    print(ypred)
    
    #mean sqarred error
    loss = sum((y_out - y_target)**2 for y_out, y_target in zip(ypred,ys))
    print("\n   The Initial Loss value:")
    print(loss)
    
    #PyTorch needed
    loss.backward()
    
    nr_of_params = len(model.parameters())
    print(f"\n   This Program has {nr_of_params} parameters in total (weights & biases)")
    print("\n   Not lets start the Gradient descent process:")
    
    # The gradient descent algorithm:
    for epoch in range(25001):
        ypred = [model(x) for x in xs] # Recalculate predictions with updated parameters
        loss = sum((y_out - y_target)**2 for y_out, y_target in zip(ypred,ys))
        
        for p in model.parameters(): # Apply of ZeroGrad after each epoch
            p.grad = 0.0
        
        loss.backward() # recalculate Grads for new loss value
        
        for p in model.parameters(): # Update params to decrease loss
            p.data += -0.4 * p.grad
            
        if epoch % 1250 == 0:
            print(f"On Epoch {epoch}: The models loss is {loss.data} and the predicted output is: {ypred}")
        
if __name__ == "__main__":
    main()