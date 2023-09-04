# Neural-Network-from-Scratch
- [View and run the code on Kaggle](https://www.kaggle.com/code/ben12002/neural-network-from-scratch/notebook)

## Overview
 - This project presents object-oriented code for creating, designing, and training neural networks for various machine learning problems.
 - All the functionality associated with neural networks is built only with NumPy.
 - To test the code, I create a neural network instance for handwritten digit classification on the [MNIST Dataset.](https://en.wikipedia.org/wiki/MNIST_database)

## Why?
When it comes to learning a specific topic, I enjoy exploring and connecting all layers of abstraction. When it comes to machine learning, this entails understanding everything from the underlying mathematics behind neural networks to how they are developed and deployed for real life problems. 

Building neural network classes from scratch (using only the assistance of NumPy), then using those classes to train a neural network, then design it on a dataset is a great way to do so.

## Usage
Viewing and running the code is as simple as visiting the [kaggle notebook link for this project](https://www.kaggle.com/code/ben12002/neural-network-from-scratch/notebook). To edit the code, just click the "edit" button on the top right corner on the aforementioned link.

Alternatively, you can download the notebook, then use it on any editor you like.

## Code Design
I designed the neural network classes, along with associated functionality, with OOP principles in mind. Specifically, I wanted the code to be easy read and to expand upon in the future. 

When it comes to the API, I wanted creating a neural network object to be as intuitive as possible:
```
net = MultiLayerPerceptron([InputLayer(units=784), DenseLayer(units=300, fn="relu"), DenseLayer(units=10, fn="softmax")],
                           weight_init="he_initialization",
                           bias_init="small_constant")
```
As shown above, creating a multilayer perceptron involves passing in an array of layer objects, followed by some optional arguments.

## Performance
I implemented additional machine learning techniques for improving the performance (i.e accuracy) of the models:
1. **Data Augmentation**: Modifying training examples in the dataset to effectively create more data that can be used to train a model. Also has been found to improve the model's ability to form accurate predictions on unseen inputs. In the case of image classification tasks, this would typically entail transforming the image, such as translation, rotation, or scaling.
2. **Learning Rate Decay**: Decreasing the learning rate according to some specific formula every epoch. Helps the model converge better.
3. **Model Checkpointing**: Saves the parameter values that led to the best model performance during training.

## MNIST Dataset
As mentioned before, I tested the code on the MNIST dataset, where I was able to achieve a 96% accuracy reliably in ~15 epochs with a single hidden layer model with 300 hidden units.
