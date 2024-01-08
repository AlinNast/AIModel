# AIModel
This is my repo where I experiment with building AI models and keep track of the evolution

# Devlog Entries:
## 1 Pytorch
Pytorch is the main tool for building AI projects BECOUSE:
 -  It provides the ability to use TENSORS ( multidimensional arrays ) just like numpy but with the superpower that they can run on the GPUs computaional power also (Pretty nice considering the new RTXs on the market) 
 - It uses Dynamic Computational Graphs, this means the graph is built on Runtime, this makes it more intuitive and allows easier debugging
 - The Autograd Module, a powerful AUTOMATIC DIFFERENTIATION tool, built in pyTorch, essential for training neural networks, as it simplifies the implementation of back propagation
 - It has its own NEURAL NETWORK MODULE, `torch.nn` provides a simple way to build neural networks
 - It has GREAT DOCUMENTATION, lots of tutorials and even PRE-TRAINED models

Even though I had python installed for other projects, it is a good practice to Create a anaconda environment and install python within it, with all ne dependencies, anaconda takes care automatically of version mismatches and generally make life easier for the developer


 Here are the booksmarks for this learning step:
 https://www.youtube.com/watch?v=ORMx45xqWkA

 https://www.youtube.com/watch?v=IC0_FRiX-sw&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=1

 https://www.youtube.com/watch?v=GvYYFloV0aA&list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b

 ## 2 Tensors

As discovered before Tensors are mutidimensional arrays, Becouse of this, they can represent different things (to be learned). In example, you need a black and white picture? 2 dimensianal matix for the pixels. Need a color picture? a 3d matix, add a dimension for the color. Have a bunch of them? another dimension for the batch, is it a movie? add another dimension for the frame and time. you get it, it may work with everything: speach, picture, music, animations. All thanks to videogames and graphic cards if you ask me.

Certain mathematical operation are possible with tensors,they have to follow certain rules, here i can summarize:

     - Element-wise Operations: Operations like addition, subtraction, multiplication, and division are performed element-wise. This means they are applied to each corresponding element of the tensors.

     - Broadcasting: When you perform operations on tensors of different shapes, PyTorch automatically 'broadcasts' the smaller tensor over the larger one. However, this only works when the shapes are compatible.

     - Matrix Multiplication: Unlike element-wise multiplication, matrix multiplication (or dot product) involves a specific calculation defined in linear algebra. In PyTorch, you use torch.matmul() for this.

     - Dimensionality: Keep an eye on the dimensions (or ranks) of your tensors. Operations might behave differently on tensors with different numbers of dimensions.

It is very likeley that I am missing on some of them, but I will figure this out while I learn more.

Most used operations:

     - Reshaping: Changing the shape of a tensor (e.g., torch.reshape()). It's crucial for preparing data and feeding it into models.

     - Slicing and Indexing: Extracting specific parts of tensors, much like you would do with arrays in Python.

     - Aggregation: Operations like sum (torch.sum()), mean (torch.mean()), and max (torch.max()) are commonly used for statistical analysis of data.

     - Element-wise Mathematical Functions: Functions like torch.exp(), torch.log(), and trigonometric functions are often used in various computations.

     - Normalization: Operations like torch.norm() for vector normalization are common, especially in preprocessing data.

Some more operation used in machine learning which i dont really understand now

Gradient Computations: Essential for training neural networks. The autograd system in PyTorch handles the computation of gradients automatically.
Weight Updates: In training loops, operations like element-wise addition and multiplication are used to update the weights of the model.
Activation Functions: Functions like ReLU (torch.relu()) are applied to the outputs of layers in a neural network.
Loss Computation: Operations like MSE (Mean Squared Error) or Cross-Entropy for calculating the loss during training.

Those are some methods used in machine learning that i assume involde or are built by the most used operation, I remmeber in hichschool that i learned these advanced algebra mathematics, like operation with matrices and trigomometry, but sadly i forgot it, I guess i will have to relearn it.

Here is the documentation for what i learned:

https://www.youtube.com/watch?v=r7QDUPb2dCM&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=2

https://www.youtube.com/watch?v=v43SlgBcZ5Y

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi


## 3 Autograd

What is Autograd? 
Autograd is an automatic differentiation system within PyTorch. It's used to automatically compute gradients—that is, the derivatives of tensors with respect to some other tensors. This is crucial for optimization in neural networks, especially during the backpropagation phase.

How is it used?
In PyTorch, you use autograd by working with torch.Tensor objects. When you create a tensor, you can set requires_grad=True to track all operations on it. After your computations, you can call .backward() on the final tensor to compute all the gradients automatically.
See autograd_test.py for examples

Why is it used?
Autograd simplifies the process of training neural networks. In a neural network, you need to update the weights based on the gradient of the loss function with respect to the weights. Autograd automates the computation of these gradients. Without autograd, you'd have to manually compute the derivative expressions for each model, which is complex and error-prone.

The loss function is a way to measure how far is the model prediction from the desired output
The weights are the parameters that determine how much influence one neuron has over another. On the bigger picture, they determine how much impact has the input over the output.

In the autograd_test.py we can see that x.grad and y.grad are not values of x and y, but they are instead the rates at which z changes with respect to changes in x and y. The z.backward() calculates those grads and storesthem in the tensor object 

The math behind obtaining these grads looks like this:

     x.grad contains the gradient of z with respect to x. In my example, z = x * x * y. To find the gradient of z with respect to x, we differentiate z with respect to x:

     ∂z/∂x=∂/∂x(x^2∗y)

     Applying the chain rule, we get:

     ∂z/∂x=2xy

     Substituting x = 2 and y = 3:

     ∂z/∂x= 2 * 2 * 3=12

     So, x.grad is tensor([12.]), which is the gradient of z with respect to x.

How to use the grad to update the weight:
The grad tells you the direction and magnitude of the steepest increase in a function.
The goal is to minimise the foss function, so the wights are ajusted in the oposite direction
The parameter learning rate is used to control how much you ajust, its usually a small number like 0.01
The basic formula for updating the wight is `new_wight = old_weight - (learning_rate * grad)`


## 4 Neural Networks:
In py torch one simple one is the linear nn, wich gets the input into a nr of neurons, that will be passed through the next layer untill it reaches the output, ReLu is typically applied between the layers, exapt the last one

Why ReLU?

Non-linearity: It introduces non-linear properties to the network, allowing it to learn more complex patterns.
Computationally Efficient: Compared to other functions like sigmoid or tanh, it's faster to compute.
Mitigates Vanishing Gradient Problem: Unlike sigmoid or tanh, its derivative is not squashed into a very small range, which helps during backpropagation.

Usefull functions in torch.nn.functional:
(Variants to ReLU actually)

Activation Functions:

F.relu: ReLU function.
F.sigmoid: Sigmoid function, used for binary classification.
F.tanh: Tanh function, similar to sigmoid but ranges from -1 to 1.
F.leaky_relu: A variant of ReLU that allows a small gradient when the unit is inactive.

Loss Functions:

F.mse_loss: Mean Squared Error, used for regression tasks.
F.cross_entropy: Cross-entropy loss, used for classification tasks.

Pooling Functions:

F.max_pool2d: Max pooling for convolutional neural networks.
F.avg_pool2d: Average pooling.

Normalization:

F.batch_norm: Batch normalization, used to stabilize and speed up training.
F.dropout: Dropout, a regularization technique to prevent overfitting.

Variants of Neural Networks

Convolutional Neural Networks (CNNs):
Purpose: Primarily used for image processing and computer vision tasks.
Key Layers: Convolutional layers (nn.Conv2d), Pooling layers (nn.MaxPool2d).
How They Work: They apply filters to the input to detect patterns like edges, textures, and other features.

Recurrent Neural Networks (RNNs) and LSTMs:
Purpose: Suited for sequential data like text, time series.
Key Layers: Recurrent layers (nn.RNN), Long Short-Term Memory layers (nn.LSTM).
How They Work: They process data sequentially, maintaining an internal state to remember past inputs.

Residual Networks (ResNets):
Purpose: Used to build very deep networks, common in advanced image processing tasks.
Key Concept: Skip connections, which allow the output of one layer to "skip" some layers and be added to the output of a later layer.
Benefit: Mitigates the vanishing gradient problem in deep networks.

Transformer Networks:
Purpose: Revolutionized natural language processing (NLP).
Key Concept: Attention mechanisms, allowing the network to focus on different parts of the input sequence for prediction.
Example: The architecture behind models like BERT, GPT.

Autoencoders:
Purpose: Used for unsupervised learning tasks like dimensionality reduction, feature learning.
Architecture: Consists of an encoder that compresses the input and a decoder that reconstructs the input from the compressed representation.
Each of these architectures serves different purposes and is suitable for various kinds of tasks. The choice depends on the nature of your data and the specific problem you're trying to solve.

The Documentation:
https://www.youtube.com/watch?v=w9U57o6wto0&t=1s
https://www.youtube.com/watch?v=NaptjtDyvuY
https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3


# 5 Micrograd

This is the first Machine learning project implemented from scratch, 

So far, The Value object is here to represent what a tensor in pytorch is, but its data is not a tensor but a single value, it stores a grad and knows the operation and the other values it came from, 

By calculating the local and general derivative, the gradient is set to the Value object

Next step was to build the neural network, first i define a Neuron

Neuron has a array o weights (Value object), array size is = to Number of inputs
array size is eqal to the number of inputs, also stores a bias, upon calling a neruon it returns: 
the sum of w*x+b.someActivationFunction() where:
w is the weight represented by one Value object
x is the input value, a array that must have the size of the number of inputs of a neuron 
b is the bias
it iterates through the array of weight and the array x at the same time


Next I define a Layer:

Layer is a Array of neurons of the size = to the desired nr of outputs
Neurons are instantieted with the size of the recieved inputs
upon call it returns a array of all the computed values of the neurons in it

Next i define the MLP

Multy Level Perceptron, it build a first layer of input neurons = to the dimensions of x, the second argument is the array coresponding to the sizes of layers, which build  neurons with the size = to previous layer

Upon call it calls every layer to compute the output, which is then used as input for the next layer untill the last