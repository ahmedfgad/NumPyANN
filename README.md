## Implementation of artificial neural networks using NumPy in addition to extraction of features and classification of the Fruits360 image dataset

This project builds artificial neural network in **Python** using **NumPy** from scratch in order to do an image classification application for the **Fruits360 dataset**. Just 4 classes are used from such a dataset which are **Apple Braeburn**, **Lemon Meyer**, **Mango**, and **Raspberry**.

At first, features are extracted from the dataset using the **extract_features.py** script. This file is expected to be located at a directory in which there are 4 folders holding the images of the 4 classes. The folders are named **apple**, **lemon**, **mango**, and **raspberry**. The script loops through the images within the 4 folders for calcualting the features which are the color histogram of the hue channel of the HSV color space. The script saves 2 files. The first one holds the features of all samples and the second one is the class labels of the samples.

After preparing the training data (inputs features and class labels), next is to implement the ANN and train it according to such data. This is done using the **ann_numpy.py** script. It is important to note that this project is divided across 2 tutorials. The first tutorial, which is implemented by this GitHub project, is titled **Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset** and available at my **LinkedIn profile** here: https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad. This tutorial just implemented the forward pass of the ANN without implementing the backward pass for updating the ANN paremters (i.e. weights). 

The second part of the project uses the **genetic algorithm (GA)** for optimizing the network weights which increases the classification accuracy. It is documented in a tutorial titled **Artificial Neural Networks Optimization using Genetic Algorithm** which will be available at my **LinkedIn profile** soon. The GitHub project implementing the second part is available at my **GitHub page** here: https://github.com/ahmedfgad/NeuralGenetic.

Everything (i.e. images and source codes) used in this tutorial, rather than the color Fruits360 images, are exclusive rights for my book cited as **"Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7 "**. The book is available at **Springer** at this link: https://springer.com/us/book/9781484241660.

The source code used in this tutorial is originally published in my **GitHub** page here: https://github.com/ahmedfgad/NumPyANN

## For contacting the author  
LinkedIn: https://www.linkedin.com/in/ahmedfgad  
Facebook: https://www.facebook.com/ahmed.f.gadd  
Twitter: https://twitter.com/ahmedfgad  
Towards Data Science: https://towardsdatascience.com/@ahmedfgad   
KDnuggets: https://kdnuggets.com/author/ahmed-gad   
E-mail: ahmed.f.gad@gmail.com
