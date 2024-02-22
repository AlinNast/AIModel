# AIModel
This is my repo where I experiment with building AI models and keep track of the evolution
From zero I noted in logs different steps of my learning process

So far it contains a Bigrad model that lears doing gradient descent on around 34,000 names, and then can generate names after learning that are as good as a probabilistic generation

The final scope of this project is to build the GPT arhitecture to work conceptually


# Devlog Entries:

### 1 Pytorch
Pytorch is the main tool for building AI projects BECOUSE:
 -  It provides the ability to use TENSORS, This complex object its the main component of the neural network:
     - It contains a multydimensioanl array to adapt to any kind of learning
     - It contains the gradient, and it can calculate it in the backpropagation process
 - It can run the complex calculation on the GPU, which is highly specialized for intense mathematical computation due to demand in video games. 
 - It uses Dynamic Computational Graphs, this means the graph is built on Runtime, this makes it more intuitive and allows easier debugging
 - The Autograd Module, Since the Tensor contains the gradient and the means to calculate it, it simplifies the implementation of back propagation
 - It has its own NEURAL NETWORK MODULE, `torch.nn` provides a simple way to build neural networks
 - It has GREAT DOCUMENTATION, lots of tutorials and even PRE-TRAINED models

Even though I had python installed for other projects, it is a good practice to Create a anaconda environment and install python within it.

Anaconda comes with all the dependencies necesary for pytorch, it takes care automatucally for version mismatches and provides the settings to run on GPU or CPU.

 Here are the booksmarks for this learning step:
 https://www.youtube.com/watch?v=ORMx45xqWkA

 https://www.youtube.com/watch?v=IC0_FRiX-sw&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=1

 https://www.youtube.com/watch?v=GvYYFloV0aA&list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b



### 2 Tensors
As discovered before, the Tensor is a object that contains:
     - A data field that is a multydimensional array 
          `Becouse of this, they can represent different things (to be learned). In example, you need a black and white picture? 2 dimensianal matix for the pixels. Need a color picture? a 3d matix`
     - A gradient field, its calulated in the gradient descent step, and its then used in learning by adjusting the data field with it
     - The tensors it came from `if aplicable` from the previous neural layer
     - The mathematical operation it came for its also stored in the object.


Certain mathematical operation are possible with tensors,they have to follow certain rules, here i can summarize:
     - Element-wise Operations: Operations like addition, subtraction, multiplication, and division are performed element-wise. This means they are applied to each corresponding element of the tensors.
     - Broadcasting: When you perform operations on tensors of different shapes, PyTorch automatically 'broadcasts' the smaller tensor over the larger one. However, this only works when the shapes are compatible.
     - Matrix Multiplication: Unlike element-wise multiplication, matrix multiplication (or dot product) involves a specific calculation defined in linear algebra. In PyTorch, you use torch.matmul() for this.
     - Dimensionality: Keep an eye on the dimensions (or ranks) of your tensors. Operations might behave differently on tensors with different numbers of dimensions.

Most used operations:
     - Reshaping: Changing the shape of a tensor (e.g., torch.reshape()). It's crucial for preparing data and feeding it into models.
          `As i know so far, it may be used in encoding the data most of the time`
     - Slicing and Indexing: Extracting specific parts of tensors, much like you would do with arrays in Python.
     - Aggregation: Operations like sum (torch.sum()), mean (torch.mean()), and max (torch.max()) are commonly used for statistical analysis of data.
     - Element-wise Mathematical Functions: Functions like torch.exp(), torch.log(), and trigonometric functions are often used in various computations.
     - Normalization: Operations like torch.norm() for vector normalization are common, especially in preprocessing data.


Here is the documentation for what i learned:

https://www.youtube.com/watch?v=r7QDUPb2dCM&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=2

https://www.youtube.com/watch?v=v43SlgBcZ5Y

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi


### 3 Autograd

What is Autograd? 
Autograd is an automatic differentiation system within PyTorch. It's used to automatically compute gradients,that is the derivatives of tensors with respect it comes from, as the tensor contains a refference to the tensors and the operation it comes from.

How is it used?
In PyTorch, you use autograd by working with torch.Tensor objects. When you create a tensor, you can set requires_grad=True to track all operations on it. After your computations, you can call .backward() on the final tensor to compute all the gradients automatically. This method is used in the gradient descent to update the tensors grads on the learning epochs
See autograd_test.py for examples

Why is it used?
Autograd simplifies the process of training neural networks. Without autograd, you'd have to manually compute the derivative expressions for each model, which is complex and error-prone.

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


### 4 Neural Networks:
In py torch one simple network is the linear nn, which gets the input into a nr of neurons, that will be passed through the next layer untill it reaches the output, ReLu is typically applied between the layers, exapt the last one

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


### 5 Micrograd

This is the first Machine learning project implemented from scratch, 

So far, The Value object is here to represent what a tensor in pytorch is, but its data is not a tensor but a single value, or a scalar. It stores a grad and knows the operation and the other values it came from. 
By calculating the local and general derivative, the gradient is set to the Value object

Next step was to build the neural network, first I defined a Neuron:
     Neuron has a array o weights (Value object), array size is = to Number of inputs (encoded input into an array or whatever datatype)
     array size is eqal to the number of inputs, also stores a bias. 
          upon calling a neruon it returns: 
          the sum of w*x+b.someActivationFunction() where:
               w is the weight represented by one Value object
               x is the input value, a array that must have the size of the number of inputs of a neuron 
               b is the bias
it iterates through the array of weight and the array x at the same time
So, it knows how many Inputs (weights) it will have an output, one computed Value


Next I define a Layer:
     Layer is a Array of neurons of the size = to the desired nr of outputs
     Neurons are instantieted with the size of the recieved inputs (previous layer size if not first)
          upon call it returns a array of all the computed values of the neurons in it


Next I define the MLP:
     Multy Level Perceptron, it builds a first layer of input neurons = to the dimensions of x (input size), the second argument is the array coresponding to the sizes of layers, which builds neurons with the size = to previous layer
          Upon call it calls every layer to compute the input, then give it to the next layer, all the way to the last layer which will output the predicted value 


Next I needed to access and adjust the weights, for this i created the params functions on the elements of the model, which returns an array (of arrays) of weights (when called in the MLP)

Whith this figured out, gradient descent can be implemented

Now the Program can be tested, with a sample array, and a expected output.
     - Instantieted the mlp 
     `These steps are the gradient descent`
          - Called it with the sample data
          - Got a predicted data
          - Calculated the MSE loss function with it and got a result
          - Reset gradients to zero (zero grad)
          - Called loss.backward and got gradients for every weight
          - Call the MLP.params to get the array of weights (p)
          - Iterate through the array with p.data += -0.01(learning step) * p.grad(the computed gradient)
     - Now all the weights have been modified to decrease the loss function and the the models prediction should be beter


### 6 Bigram model

The next logical step is to transform what I learned with numbers in a text based program, a simple one is to build a name generator with a birgram model

A bigram model knows one leter and outputs the next one

In order to do so, first I need to transform text into numbers, make the machine work with numbers and then transform them back to text

I created a Dictionary with the characters and a index (alphabetically ordered)
Using the dictioary, and a iteration through every character of the string of training data, the tensor (N) now maps how many times a characters is followed by another for every character in the array

Since now we have the count of every letter following another the name generator is build:

     First step:
          Starting on the first row (first letters following dot), normalizing the values and transforming them in probabilies
          get one letter acording to its probability

     Second Step:
          With the new generated letter, its row is now selected and the normalizing and prob transform is now applied
          the next letter is generated acording to the probability

     Last step:
          If the genereted char is the end char '.' the loop stops and returns a value

     Now the algoritm can generate a name probabilisticly 

Evaluating the word. 
In order to go to train a model I first need to figure out a way to evaualte its untrained predictions. The solution for this is a negative Logaritmic likelihood, the formula can be seen in the code. It sum up the normalized values of the probabilities (normalized by log function), and its negative in order to return a positive float

In order to recreate something like the tensor N that also implements the machine learning:
     I create a tensor W of the same shape as N but with normalized random values
     I encode the data (string of name) into a tensor of right shape with the one_hot() function
     I multiply the W with enc to get a new probabilities tensor (the output)
     On this output tensor, the same loss function is applied
     Gradient descent is applied on the W tensor
     Process is repreated a few epochs, printing the loss shows it gets closer to the loss of the probabilities tensor N
Generating a Name before and after the gradient descent, and also with the probabilities tensor N shows that the model is learning to predict the next letter based on the training data


### 7 MLP

On this experiment I combined the encoding from the previous model, turning the list of names into integers but instead of feeding them into a single layer to obtain the predictions I created a MLP, first encoding layer, the next hidden layer, a final layer and loss function calculated by a cross entropy.

#### Steps of implementations:
1. First i built the vocabulary, just as the previous project
2. Then i built the dataset:
     - For this project I implemented the segregation of the data into train/dev/test this is similar to what a real production project looks like
3. I Built the embeding layer:
     - Just as the previous project, this layer is built by generating a tensor with random normalized values
     - For the size of the embeding layer I chose (27, 10), 27 stands for the size of the vocabulary, and 10 is a arbitrary value I chose and even changed to see how the model behaves
4. I build the Hidden Layer:
     - Again tensor with random normalized values
     - The size of this layer is 30, 100
     - the first size 30 is the last size of the previous layer '10' times the block size which is 3 in this model (blocksize means how many characters the model take into accounf for generating one)
     - the second size 100 again is arbitrary and i ajusted it to see how the loss value changes
5. The Final Layer:
     -  Size 100 27
     -  This one has the size of 100 as to match the output of the previous layer
     -  27 to match the size of the vocabulary tensor
6. Before building the forward function I split the dataset into minibatches:
     -  Just as production project use minibatches, this was a thechnique to learn
     - By generating a random index tensor with value between the 0 and max size of dataset
     - Now the embeding data tensor size is smaller
     - The trainig epochs are much faster and the training still occurs on all the training dataset
7. The forward pass, after the training dataset is selected in minibathces:
     - the embeded data ~ First Layer is activated by a tanh function into the next layer
     - tanh( emb * hidden layer + bias)
     - in order for the matmul to work the emb need to be concatenated, there are many ways to do it
     - The Hidden Layer is activated by matmult with the last layer giving the output logits
     - The loss functions is calculated between the output logits and the desired output with cross entropy
8. The backward pass is more straightforward:
     - Clearing the gradinets
     - calculate the derivatives of the loss operation
     - Adjust the weights and biases with the new gradinets
     - For this model a dynamic learning rate was used in order to learn how production models are trained
9. Evaluation:
     - the loss function is calculater again with the dev data set in order to proove training works


#### New aditions:
- Spliting the dataset into training, dev and test
- Using a dynamic learning step
- Using cross entropy as a loss calculation
- Testing the loss function with a separate dataset
- Using a block of 3 character to predict the next one rather than just one


### MLP v2

It is a Upgraded MLP, with a bigger number of parameters, with more training epochs, normalized parameters to avoid inactive neurons, and batch normalization to support deeper neural networks.

#### Steps of implementation:

1. I started by recreating the previous MLP but with leaving asside some hard coded values to allow for easier modification of the ANN, also making certain functions more easily accessible. This is the initial ANN:

          MLP Initialized
          Layers created in 0.000819200009573251 sec
          This MLP contain 11897 parameters
          Value of the loss funtion: 30.54128646850586 on training step:0
          Value of the loss funtion: 2.9965057373046875 on training step:10000
          Value of the loss funtion: 2.173659086227417 on training step:20000
          Training of 20001 steps completed 22.84705560002476 sec
          Evaluation loss: 2.3344485759735107

2. The initial loss is obviously very high so a logical step is to reduce the initial loss in order to make the NeuralNet beter, I did this by reducing the initial biases of the output layer b2 to 0 and the weights of the output layer W2 with 0.01, this way reducing a lot of the default entropy caused by the randomized normal tensor generation, and the output is here:

          MLP Initialized
          Layers created in 0.0014186000335030258 sec
          This MLP contain 11897 parameters
          Value of the loss funtion: 3.294851779937744 on training step:0
          Value of the loss funtion: 1.9210432767868042 on training step:10000
          Value of the loss funtion: 2.1803672313690186 on training step:20000
          Training of 20001 steps completed 22.89989629999036 sec
          Evaluation loss: 2.1928908824920654
          sari.

3. Already some improvement, next step is to look at the hidden layer activation, The way Tanh works in the backprobagation can be ellusive, the derivative of tanh(x) = 1 - tanh**2(x) this means that if a neuron will have a value close to 1 or -1, in the loss.backward() phase its gradient will be calculated as nearly 0, which makes it hard to train, even inactive, and this is undesired.more about the tanh function here: https://fr.wikipedia.org/wiki/Tangente_hyperbolique. To fix this. the same technique will be applied to the hidden layer, W1 * 0.1 and b * 0.1. The goal is to reduce high values not to remove all the entropy, the results are here:

          MLP Initialized
          Layers created in 0.001290299987886101 sec
          This MLP contain 11897 parameters
          Value of the loss funtion: 3.29561710357666 on training step:0
          Value of the loss funtion: 2.240442991256714 on training step:10000
          Value of the loss funtion: 1.8903627395629883 on training step:20000
          Training of 20001 steps completed 21.337516199972015 sec
          Evaluation loss: 2.228701114654541
          chackry.

4. After some more research I found out that a scientific paper published bi Kaiming, actually looked into the best ways to scale a neuron weights relative to the activation function: https://arxiv.org/abs/1502.01852 . According to the paper instead of multiplying W1 with some trial an error values, I used (5/3)/(n**0.5) where n is the layers first dimension

          MLP Initialized
          Layers created in 0.0011490000179037452 sec
          This MLP contain 11897 parameters
          Value of the loss funtion: 3.2947452068328857 on training step:0
          Value of the loss funtion: 2.152228355407715 on training step:10000
          Value of the loss funtion: 2.002431869506836 on training step:20000
          Training of 20001 steps completed 23.656168000015896 sec
          Evaluation loss: 2.2089436054229736
          molanita.

5. Batch Normalization, the next big breakthrough in machine learning, The scientific paper describing it is linked here: https://arxiv.org/abs/1502.03167, Its implementation here is mostly demonstrative due to the size of the MLP it wont impact the performance too much, but on big ANN they play a important role. By using a Normalization layer that also keeps track of the mean and stable distributionit maintains the values of the preactivatted layer to a Gaussian distribution, this also prevents over fitting data

          MLP Initialized
          Layers created in 0.02382140001282096 sec
          This MLP contain 12297 parameters        
          Value of the loss funtion: 3.2917559146881104 on training step:0
          Value of the loss funtion: 2.179684638977051 on training step:10000
          Value of the loss funtion: 2.0975732803344727 on training step:20000
          Training of 20001 steps completed 27.136896900017746 sec
          Evaluation loss: 2.236245632171631
          daviah.




#### New aditions:
 - Reduced initial loss due to descaling the output layer W and b
 - Eficiently reduce loss and increase training efficiency by scaling the Hidden layer with a Kaiming ratio
 - Introduced batch normalization, it doesent improve performance too much but conceptually is essential for deep neural networks


 ### WaveNet 

 Its Like a MLP but with a Deeper NN, more layers basically, 

 #### Steps of implementations:
 1. It starts with the previous MLP code, but getting rid of all the hard values and functions and building the layers as classes in order to have more versatility
 2. Apply the same logic on the embeding layer with the same principles of pytorch, and encapsulate everything into a model object.
 3. In order to apply the waveNet structure to this ANN the block size needs to increase, with a block size of 8 instead of a 3 training loss decreases faster, but the limit is set by introducing all the information of the embedding into a single layer of neurons