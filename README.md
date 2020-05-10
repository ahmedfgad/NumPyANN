## NumpyANN: Implementing Artificial Neural Networks using NumPy

**NumPyANN** is an open-source project for building artificial neural networks in **Python 3** using **NumPy** from scratch. The main module of this project is the `nn.py` module which builds the network layers, implements the activations functions, trains the network, makes predictions, and more. 

The purpose of this project is to only implement the **forward pass** of a neural network without using a training algorithm.

For training a neural network using the genetic algorithm, check this project (https://github.com/ahmedfgad/NeuralGenetic) in which the genetic algorithm is used for training the network.

Feel free to leave an issue in this project (https://github.com/ahmedfgad/NumPyANN) in case something is not working properly or to ask for questions. I am also available for e-mails at ahmed.f.gad@gmail.com

## Supported Layers

Up to this time, the supported layers of the project are:

1. Input: Implemented using the `nn.InputLayer` class.

2. Dense (Fully Connected): Implemented using the `nn.DenseLayer` class.

In the future, more layers will be added.

## Supported Activation Functions

The supported activation functions are:

1. Sigmoid: Implemented using the `nn.sigmoid()` function.

2. Rectified Linear Unit (ReLU): Implemented using the `nn.relu()` function.

## Steps to Build a Neural Network

The `example.py` file has an example of building and using a neural network using this project.

### Reading the Data

Before building the network architecture, the first thing to do is to prepare the data that will be used for training the network. 

Besides the Python files, there are 2 files within the project holding data extracted from 4 classes of the **Fruits360** dataset which are **Apple Braeburn**, **Lemon Meyer**, **Mango**, and **Raspberry**. The project has 4 folders holding the images for the 4 classes.

The 2 files are named:

1. **dataset_features.npy**: The features.
2. **outputs.npy**: The outputs.

There is a Python script named **extract_features.py** which reads the raw images of the dataset, prepare the features and the outputs as NumPy array, and saves the arrays in the previous 2 files. 

This script loops through the images within the 4 folders for calculating the features (color histogram of the hue channel of the HSV color space). 

After the 2 files are created, then just read them to return the NumPy arrays according to the next 2 lines:

```python
data_inputs = numpy.load("dataset_features.npy")
data_outputs = numpy.load("outputs.npy")
```

After the data is prepared, next is to created the network architecture.

### Building the Network Architecture

Each layer supported by the project has class by which a new layer can be created. For example, an input layer can be created by instantiating the `nn.InputLayer` class according to the next code. The constructor of this class accepts a single parameter specifying the number of input neurons (i.e. length of the feature vector). A network can only has a single input neuron.

```python
num_inputs = data_inputs.shape[1]

input_layer = nn.InputLayer(num_inputs)
```

Following the input layer, other layers can be added by instantiating the `nn.DenseLayer` class. The class constructor accepts the following parameters:

- `num_neurons`: Number of neurons.
- `previous_layer`: A reference to the layer preceding the current layer in the network architecture.
- `activation_function="sigmoid"`: The type of the activation function which defaults to `sigmoid`.

The constructor of the `nn.DenseLayer` class is responsible for initializing the weights of each layer.

Here is an example of creating 2 dense layers. Normally, the last dense layer is regarded as the output layer.

```python
hidden_layer = nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=input_layer, activation_function="relu")
output_layer = nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer2, activation_function="sigmoid")
```

After the data is prepared and the network architecture is created, next is to train the network.

### Training the Network

Even no training algorithm is used in this project, the weights are updated using the learning rate according to the next equation. It is not the best way to update the weights but it is better than keeping it as it is by making some small changes to the weights.

```python
new_weights = weights - learning_rate * weights
```

To train the network, the `nn.train_network()` function is used. The parameters accepted by the function are:

- `num_epochs`: Number of epochs.
- `last_layer`: Last layer in the network architecture.
- `data_inputs`: Data features.
- `data_outputs`: Data outputs.
- `learning_rate`: Learning rate.

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

The `nn.predict_outputs()` function uses the trained network for making predictions. It accepts the following parameters:

- last_layer: The last layer in the network architecture.
- data_inputs: Data features.

Here is an example.

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