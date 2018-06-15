#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

# importing libraries
import ThreeStateGenerator as tsg
import parse
import os
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
import math
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from keras import regularizers
from keras.layers import Dropout
from numpy import array
from sklearn.metrics import confusion_matrix
import sys

# Associating each character with a number (Encoding the characters for the model)
ACIDS = {'<': 1, '>': 2, 'A': 3, 'C': 4, 'E': 5, 'D': 6, 'G': 7, 'F': 8, 'I': 9, 'H': 10, 'K': 11, 'M': 12, 'L': 13, 'N':14, 'Q':15, 'P':16, 'S':17, 'R':18, 'T':19, 'W':20, 'V':21, 'Y':22, 'X':23, 'U': 24, 'Z': 25, 'B':26, 'O':27}
LABELS = {'<': 1, '>': 2, '?': 2, 'S': 3, ' ': 4, 'H': 5}


def encode_sequences(string, vocab_dict):
    """
    Performs one-hot encoding.
    
    string: input string from the input data
    vocab_dict: dictionary of the amino acid or label character encoded vocabulary
    
    Output:
        sequence: one-hot encoded sequence
    """
    
    sequence = []
    
    vec_length = np.max(list(vocab_dict.values()))
    
    for i in range(len(string)):
        temp = np.zeros(vec_length)
        temp[vocab_dict[string[i]]-1] = 1
        sequence.append(temp)
    
    return sequence


def pad_sequence(input_data, win_size):
    """
    Helper function to pad the input sequences.
    
    input_data: input data given by the parser
    win_size: window size / desired Kmer length
    
    Output:
        padded_sequence: sequence padded with zeros
    """
    
    padded_sequence = []
    for i in range(len(input_data)):
        padded_sequence.append(np.pad(input_data[i], math.ceil(win_size/2), 'constant'))
    
    return padded_sequence

def kmerify(input_data, output_data, win_size):
    """
    Pads the input sequences, then returns kmers and the corresponding labels.
    
    input_data: the protein list returned by csvify
    output_data: the list of secondary structures returned by csvify
    win_size: the length of the kmers
    
    Output:
        KmerX: a numpy array of all the win_size-long protein fragments
        labelY: a list of the labels corresponding to the middle amino acid 
                in each kmer
    """
    
    KmerX = []
    labelY = []
    
    # padding the input sequences
    paddedX = pad_sequence(input_data, win_size)
    
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            kmer = paddedX[i][j:j+win_size]
            KmerX.append(np.array(kmer))
            labelY.append(output_data[i][j])
    return np.array(KmerX), labelY

"""
def the_nn():
    num_nodes = 140
    num_layers = 2
    model = Sequential()
    model.add(Dense(num_nodes, activation = 'relu', 
                    input_shape = (X.shape[1], X.shape[2]),
                    kernel_regularizer = None))
    model.add(Flatten())
    for l in range(num_layers-1):
        model.add(Dense(num_nodes, activation = 'relu',
                    kernel_regularizer = None))
    model.add(Dense(len(Y[1]), activation='softmax',
                    kernel_regularizer = None))
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = 'sgd',
                  metrics = ['accuracy'])
    
    return model
    
"""

def the_nn(f,X, Y, num_nodes = 110, num_layers = 2, 
              activation = 'relu', ouput_activation = 'softmax',
              batch_size = 50, epochs = 50, classweight = None,
              validation_split = 0.2, optimizer = "adam",
              kernel_regularizer = regularizers.l2(0.01)):
    
    """
    Creates and fits a neural network.
    
    X: input data
    Y: output data (labels)
    num_nodes: the number of nodes in the layers (default: 140)
    activation: activation function (default: relu)
    output activation: activation function in the output layer (default: softmax)
    batch_size: the batch size used during fitting (default: 100)
    epochs: number of epochs run during fitting (default: 50)
    class_weight: class weight to remove class imbalance (default: None)
    validation_split: fraction of data used for validation
    optimizer: model optimizer used during compiling
    kernel_regularizer: regularizer used in the layers (default: None)
    
    Output:
        accuracy and model loss plots 
    
    """
    
    model = Sequential()
    model.add(Dense(num_nodes, activation = activation, 
                    input_shape = (X.shape[1], X.shape[2]),
                    kernel_regularizer = None))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    for l in range(num_layers-1):
        model.add(Dense(num_nodes, activation = activation,
                    kernel_regularizer = kernel_regularizer))
        #model.add(Dropout(0.2))
    model.add(Dense(len(Y[1]), activation='softmax',
                    kernel_regularizer = None))
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    
    
    history = model.fit(X, Y, validation_split = validation_split, 
                        epochs = epochs, batch_size = batch_size)
    

    Xnew = f
    Ynew = '>' * len(Xnew[0])
    lst = []
    lst.append(Ynew)

    hotX = []
    hotY = []
    for i in range(len(Xnew)):
        hotX.append(encode_sequences(Xnew[i], ACIDS))
        hotY.append(encode_sequences(lst[i], LABELS))
    x, y = kmerify(hotX, hotY, 17)
    x = np.array(x)
    

    output = model.predict(x)

    return output,history

def jNN(f):
   # getting the files and generating 3 states
    path = os.getcwd()
    path = os.path.join(path,"DATA")
    path_1 = os.chdir(path)

    X_strings, Y_strings,file_lst = parse.parse_file(path_1)
    Y_strings = tsg.replace_all(Y_strings)
        
    # one hot encoding
    onehotX = []
    onehotY = []
    for i in range(len(X_strings)):
        onehotX.append(encode_sequences(X_strings[i], ACIDS))
        onehotY.append(encode_sequences(Y_strings[i], LABELS))
    
    # getting Kmers
    KmerX, labelY = kmerify(onehotX, onehotY, 17)
    
    # the inputs
    X = np.array(KmerX)
    Y = np.array(labelY)
    
    # removing class imbalance
    classweight = class_weight.compute_class_weight('balanced', 
                  np.unique([np.argmax(i) for i in list(Y)]), 
                  [np.argmax(i) for i in list(Y)])
    

    
    output,history = the_nn(f,X, Y, num_nodes = 110, num_layers = 2, 
                 activation = 'relu', ouput_activation = 'softmax',
                 batch_size = 50, epochs = 50, classweight = classweight,
                 validation_split = 0.2, optimizer = "adam", kernel_regularizer = regularizers.l1(0.01))
    
    INV_LABELS = {}
    for key in LABELS.keys():
        INV_LABELS[LABELS[key]] = key
    prediction = ''.join([INV_LABELS[np.argmax(output[i])+1] for i in range(output.shape[0])])
    print("prediction : ",prediction)

    return prediction
