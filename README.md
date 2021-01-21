# Monkey Classifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/urastogi885/monkey-classifier/blob/main/LICENSE)

## Overview
In this project, we perform monkey species classification using convolutional neural networks (CNNs). We use the [10 
Monkey Species Dataset](https://drive.google.com/file/d/1vKX96-7weV_Ex_e4_GQ1DH5ONw02re7m/view?usp=sharing). I have 
used my [digit recognition CNN](https://github.com/urastogi885/mnist-digit-recognition/blob/main/CNN.m) as a base for 
the custom CNN. It has been updated to include 2 more 2D convolutional layers making it a 4 convolutional layer network.
The base network for transfer learning is VGG16. The VGG model has been appended with 2 fully-connected layers and a 
dropout layer. Apart from that, the last 5 layers of the VGG model were made trainable to get better generalization.

## Dependencies
- Python 3.8
- Pandas 1.0.4
- Keras 2.4.3
- Tensorflow 2.4.0

Stick to these versions of dependencies, especially Keras and Tensorflow, if you want to directly use the scripts.

## Run
- Run the custom 4-layer network and transfer learning network using Python.
- Note if you do not have GPU properly setup with Tensorflow, the network would take hours to train. With GPU, training
  would take around 20 minutes.
- Open terminal, head to the project directory, and run the following python scripts:
```
cd <project directory/Code>
python my_cnn.py
python transfer_learning.py
```
- Note that you might have to use your alias for Python-3
