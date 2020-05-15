## NumpyANN: Implementing Artificial Neural Networks using NumPy

As part of a series of 3 projects that uses Python 3 (with the user of NumPy) to build and train artificial neural networks (ANNs) using the genetic algorithm (GA), **NumPyANN** is the second project in the series that builds artificial neural networks in **Python 3** using **NumPy** from scratch. The purpose of this project is to only implement the **forward pass** of a neural network without using a training algorithm. Currently, it only supports classification and later regression will be also supported.

The main module of this project is the `nn.py` module which builds the network layers, implements the activations functions, trains the network, makes predictions, and more. 

**IMPORTANT** *If you are coming for the tutorial code, then it has been moved to the [TutorialProject](https://github.com/ahmedfgad/NumPyANN/tree/master/TutorialProject) directory on 10 May 2020.*

## The Projects in the Series

The 3 projects in the series of building and training neural networks using the genetic algorithm are as follows:

1. [GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython): Implements the genetic algorithm.
2. [NumPyANN](https://github.com/ahmedfgad/NumPyANN): Implements neural networks without being trained (i.e. only the forward pass).
3. [NeuralGenetic](https://github.com/ahmedfgad/NeuralGenetic/): Trains neural networks implemented in [NumPyANN](https://github.com/ahmedfgad/NumPyANN) using the genetic algorithm implemented in [GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython).

Feel free to leave an issue in this project (https://github.com/ahmedfgad/NumPyANN) in case something is not working properly or to ask for questions. I am also available for e-mails at ahmed.f.gad@gmail.com

## Supported Layers

Up to this time, the supported layers of the project are:

1. **Input**: Implemented using the `nn.InputLayer` class.

2. **Dense** (Fully Connected): Implemented using the `nn.DenseLayer` class.

In the future, more layers will be added. The next subsections discuss the layers.

### `nn.InputLayer` Class

The `nn.InputLayer` class creates the input layer for the neural network. For each network, there is only a single input layer. This class has no methods or class attributes. All it has is a constructor that accepts a parameter named `num_neurons` representing the number of neurons in the input layer. An instance attribute named `num_neurons` is created within the constructor to keep such a number. Here is an example of building an input layer with 20 neurons.

```python
input_layer = nn.InputLayer(num_neurons=20)
```

Here is how the single attribute `num_neurons` within the instance of the `nn.InputLayer` class can be accessed.

```python
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

This is everything about the input layer.

### `nn.DenseLayer` Class

Using the `nn.DenseLayer` class, dense (fully-connected) layers can be created. To create a dense layer, just create a new instance of the class. The constructor accepts the following parameters:

- `num_neurons`: Number of neurons in the dense layer.
- `previous_layer`: A reference to the previous layer. Using the `previous_layer` attribute, a linked list is created that connects all network layers.
- `activation_function`: A string representing the activation function to be used in this layer. Defaults to `"sigmoid"`. Currently, the supported activation functions are `"sigmoid"` and `"relu"`.

Within the constructor, the accepted parameters are used as instance attributes. Besides the parameters, some new instance attributes are created which are:

- `initial_weights`: The initial weights for the dense layer.
- `trained_weights`: The trained weights of the dense layer. This attribute is initialized by the value in the `initial_weights` attribute.

Here is an example for creating a dense layer with 12 neurons. Note that the `previous_layer` parameter is assigned to the previous created input layer `input_layer`. 

```python
dense_layer = nn.DenseLayer(num_neurons=12,
                            previous_layer=input_layer,
                            activation_function="relu")
```

Here is how to access some attributes in the dense layer:

```python
num_dense_neurons = dense_layer.num_neurons
dense_initail_weights = dense_layer.initial_weights

print("Number of dense layer attributes =", num_dense_neurons)
print("Initial weights of the dense layer :", dense_initail_weights)
```

Because `dense_layer` holds a reference to the input layer, then the number of input neurons can be accessed.

```python
input_layer = dense_layer.previous_layer
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

Here is another dense layer. This dense layer's `previous_layer` attribute points to the previously created dense layer.

```python
dense_layer2 = nn.DenseLayer(num_neurons=5,
                             previous_layer=dense_layer,
                             activation_function="relu")
```

Because `dense_layer2` holds a reference to `dense_layer` in its `previous_layer` attribute, then the number of neurons in `dense_layer` can be accessed.

```python
dense_layer = dense_layer2.previous_layer
dense_layer_neurons = dense_layer.num_neurons

print("Number of dense neurons =", num_input_neurons)
```

After getting the reference to `dense_layer`, we can use it to access the number of input neurons.

```python
dense_layer = dense_layer2.previous_layer
input_layer = dense_layer.previous_layer
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

Assuming that `dense_layer2` is the last dense layer, then it is regarded as the output layer.

#### `previous_layer` Attribute

The `previous_layer` attribute in the `nn.DenseLayer` class creates a linked list between all the layers in the network architecture as described by the next figure. 

The last (output) layer indexed N points to layer **N-1**, layer **N-1** points to the layer **N-2**, the layer **N-2** points to the layer **N-3**, and so on until reaching the end of the linked list which is layer 1 (input layer).

![Layers Linked List](https://user-images.githubusercontent.com/16560492/81918975-816af880-95d7-11ea-83e3-34d14c3316db.jpg)

The linked list allows returning all properties of all layers in the network architecture by just passing the last layer in the network. Using the `previous_layer` attribute of layer **N**, the layer **N-1** can be accessed. Using the `previous_layer` attribute of layer **N-1**, layer **N-2** can be accessed. The process continues until reaching a layer that does not have a `previous_layer` attribute (which is the input layer).

The properties include the weights (initial or trained), activation functions, and more. Here is how a `while` loop is used to iterate through all the layers. The `while` loop stops only when the current layer does not have a `previous_layer` attribute. This layer is the input layer.

```python
layer = dense_layer2

while "previous_layer" in layer.__init__.__code__.co_varnames:
    print("Number of neurons =", layer.num_neurons)

    # Go to the previous layer.
    layer = layer.previous_layer
```

## Functions to Manipulate Neural Networks

There are a number of functions existing in the `nn.py` module that help to manipulate the neural network.

### `layers_weights(last_layer, initial=True)`

Creates and returns a list holding the weights matrices of all layers in the neural network.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `initial`: When `True` (default), the function returns the **initial** weights of the layers using the layers' `initial_weights` attribute. When `False`, it returns the **trained** weights of the layers using the layers' `trained_weights` attribute. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, either the initial weights or the trained weights are returned based on where the `initial` parameter is `True` or `False`.

### `layers_weights_as_vector(last_layer, initial=True)`

Creates and returns a list holding the weights **vectors** of all layers in the neural network. The weights array of each layer is reshaped to get a vector.

This function is similar to the `layers_weights()` function except that it returns the weights of each layer as a vector, not as an array. 

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `initial`: When `True` (default), the function returns the **initial** weights of the layers using the layers' `initial_weights` attribute. When `False`, it returns the **trained** weights of the layers using the layers' `trained_weights` attribute. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, either the initial weights or the trained weights are returned based on where the `initial` parameter is `True` or `False`.

### `layers_weights_as_matrix(last_layer, vector_weights)`

Converts the network weights from vectors to matrices.

Compared to the `layers_weights_as_vectors()` function that only accepts a reference to the last layer and returns the network weights as vectors, this function accepts a reference to the last layer in addition to a list holding the weights as vectors. Such vectors are converted into matrices.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `vector_weights`: The network weights as vectors where the weights of each layer form a single vector.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, the shape of its weights array is returned. This shape is used to reshape the weights vector of the layer into a matrix. 

### `layers_activations(last_layer)`

Creates and returns a list holding the names of the activation functions of all layers in the neural network.

Accepts the following parameter:

- `last_layer`: A reference to the last (output) layer in the network architecture.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, the name of the activation function used is returned using the layer's `activation_function` attribute. 

### `sigmoid(sop)`

Applies the sigmoid function and returns its result.

Accepts the following parameters:

* `sop`: The input to which the sigmoid function is applied.

### `relu(sop)`

Applies the rectified linear unit (ReLU) function and returns its result.

Accepts the following parameters:

* `sop`: The input to which the sigmoid function is applied.

### `train_network(num_epochs, last_layer, data_inputs, data_outputs, learning_rate)`

Trains the neural network.

Accepts the following parameters:

- `num_epochs`: Number of epochs.
- `last_layer`: Reference to the last (output) layer in the network architecture.
- `data_inputs`: Data features.
- `data_outputs`: Data outputs.
- `learning_rate`: Learning rate.

For each epoch, all the data samples are fed to the network to return their predictions. After each epoch, the weights are updated using only the learning rate. No learning algorithm is used because the purpose of this project is to only build the forward pass of training a neural network.

### `update_weights(weights, network_error, learning_rate)`

Calculates and returns the updated weights. Even no training algorithm is used in this project, the weights are updated using the learning rate. It is not the best way to update the weights but it is better than keeping it as it is by making some small changes to the weights.

Accepts the following parameters:

- `weights`: The current weights of the network.
- `network_error`: The network error.
- `learning_rate`: The learning rate.

### `update_layers_trained_weights(last_layer, final_weights)`

After the network weights are trained, this function updates the `trained_weights` attribute of each layer by the weights calculated after passing all the epochs (such weights are passed in the `final_weights` parameter)

By just passing a reference to the last layer in the network (i.e. output layer) in addition to the final weights, this function updates the `trained_weights` attribute of all layers.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `final_weights`: An array of weights of all layers in the network after passing through all the epochs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, its `trained_weights` attribute is assigned the weights of the layer from the `final_weights` parameter. 

### `predict_outputs(last_layer, data_inputs)`

Uses the trained weights for predicting the samples' outputs. It returns a list of the predicted outputs for all samples.

Accepts the following parameters:

* `last_layer`: A reference to the last (output) layer in the network architecture.

* `data_inputs`: Data features.

All the data samples are fed to the network to return their predictions. 

## Helper Functions

There are a number of helper functions in the `nn.py` module that do not directly manipulate the neural networks.

### `to_vector(array)`

Converts a passed NumPy array (of any dimensionality) to its `array`  parameter into a 1D vector and returns the vector.

Accepts the following parameters:

* `array`: The NumPy array to be converted into a 1D vector.

### `to_array(vector, shape)`

Converts a passed vector to its `vector`  parameter into a a NumPy array and returns the array.

Accepts the following parameters:

- `vector`: The 1D vector to be converted into an array.
- `shape`: The target shape of the array.

## Supported Activation Functions

The supported activation functions are:

1. Sigmoid: Implemented using the `nn.sigmoid()` function.

2. Rectified Linear Unit (ReLU): Implemented using the `nn.relu()` function.

## Steps to Build a Neural Network

The `example.py` file has an example of building and using a neural network using this project. The steps are:

- Reading the Data
- Building the Network Architecture
- Training the Network
- Making Predictions

### Reading the Data

Before building the network architecture, the first thing to do is to prepare the data that will be used for training the network. 

Besides the Python files, there are 2 files within the project holding data extracted from 4 classes of the **Fruits360** dataset which are **Apple Braeburn**, **Lemon Meyer**, **Mango**, and **Raspberry**. The project has 4 folders holding the images for the 4 classes.

The 2 files are named:

1. **dataset_features.npy**: The features.
2. **outputs.npy**: The outputs.

There is a Python script named **extract_features.py** which reads the raw images of the dataset, prepares the features and the outputs as NumPy array, and saves the arrays in the previous 2 files. 

This script loops through the images within the 4 folders for calculating the features (color histogram of the hue channel of the HSV color space). 

After the 2 files are created, then just read them to return the NumPy arrays according to the next 2 lines:

```python
data_inputs = numpy.load("dataset_features.npy")
data_outputs = numpy.load("outputs.npy")
```

After the data is prepared, next is to create the network architecture.

### Building the Network Architecture

The input layer is created by instantiating the `nn.InputLayer` class according to the next code. A network can only have a single input neuron.

```python
num_inputs = data_inputs.shape[1]

input_layer = nn.InputLayer(num_inputs)
```

After the input layer is created, next is to create a number of dense layers according to the next code. Normally, the last dense layer is regarded as the output layer.

```python
hidden_layer = nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=input_layer, activation_function="relu")
output_layer = nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer2, activation_function="sigmoid")
```

After both the data and the network architecture are prepared, the next step is to train the network.

### Training the Network

Here is an example of using the `nn.train_network()` function.

```python
nn.train_network(num_epochs=10,
                 last_layer=output_layer,
                 data_inputs=data_inputs,
                 data_outputs=data_outputs,
                 learning_rate=0.01)
```

After training the network, the next step is to make predictions.

### Making Predictions

The `nn.predict_outputs()` function uses the trained network for making predictions. Here is an example.

```python
predictions = nn.predict_outputs(last_layer=output_layer, data_inputs=data_inputs)
```

It is not expected to have high accuracy in the predictions because no training algorithm is used. 

Please check the `example.py` file which creates an example of building a network using this project.

## Further Reading

On **10 May 2020**, the project was updated by making a major change to the original code which is using object-oriented programming for creating the layers. The original code is available in the [**TutorialProject**](https://github.com/ahmedfgad/NumPyANN/tree/master/TutorialProject) directory of the GitHub project: https://github.com/ahmedfgad/NumPyANN/tree/master/TutorialProject. 

The [original code](https://github.com/ahmedfgad/NumPyANN/tree/master/TutorialProject) is implemented in a tutorial titled [**Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset**](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad) which is available at these links:

- https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad
- https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491
- https://www.kdnuggets.com/2019/02/artificial-neural-network-implementation-using-numpy-and-image-classification.html

The tutorial is still useful for understanding the recent implementation.

For more information about neural networks and get started with deep learning for computer vision, check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665). 

## Training Neural Networks using the Genetic Algorithm

There is an extension to the project for using the **genetic algorithm (GA)** for training the network. After the network is trained using the GA, the classification accuracy increased. 

At first, the genetic algorithm implementation is available in this GitHub project: https://github.com/ahmedfgad/GeneticAlgorithmPython. Out of this project, a library named PyGAD is made available at [PyPi](https://pypi.org/project/pygad): https://pypi.org/project/pygad. Just install it using pip:

```python
pip install pygad
```

The project that trains neural networks with the genetic algorithm is available at this GitHub page: https://github.com/ahmedfgad/NeuralGenetic. This project is documented in a tutorial titled [**Artificial Neural Networks Optimization using Genetic Algorithm**](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad). Find the tutorial at these links:

- https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad
- https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e
- https://www.kdnuggets.com/2019/03/artificial-neural-networks-optimization-genetic-algorithm-python.html

## For contacting the author  

- E-mail: ahmed.f.gad@gmail.com
- LinkedIn: https://www.linkedin.com/in/ahmedfgad  
- Facebook: https://www.facebook.com/ahmed.f.gadd  
- Twitter: https://twitter.com/ahmedfgad  
- Amazon Author Page: https://amazon.com/author/ahmedgad
- Paperspace: https://blog.paperspace.com/author/ahmed
- Heartbeat: https://heartbeat.fritz.ai/@ahmedfgad
- Towards Data Science: https://towardsdatascience.com/@ahmedfgad   
- KDnuggets: https://kdnuggets.com/author/ahmed-gad
