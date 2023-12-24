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