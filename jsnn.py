#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

# import tensorflow as tf
import numpy as np
import tqdm
import MSA_parser
import parse

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Dropout
from keras.layers import LSTM, GRU, Masking, Embedding, SimpleRNN
from keras.optimizers import SGD, RMSprop, Adam
from keras import initializers

from sklearn.utils import class_weight
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras import regularizers
import ThreeStateGenerator
import os

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Associating each character with a number (Encoding the characters for the model)
ACIDS = {'<': 1, '>': 2, 'A': 3, 'C': 4, 'E': 5, 'D': 6, 'G': 7, 'F': 8, 'I': 9, 'H': 10, 'K': 11, 'M': 12, 'L': 13, 'N':14, 'Q':15, 'P':16, 'S':17, 'R':18, 'T':19, 'W':20, 'V':21, 'Y':22, 'X':23, 'U': 24, 'Z': 25, 'B':26, 'O':27}
#LABELS = {'<': 1, '>': 2, 'NoSeq': 2, 'L': 3, ' ':3, 'B': 4, 'E': 5, 'G': 6, 'I': 7, 'H': 8, 'S': 9, 'T': 10}
LABELS = {'<': 1, '>': 2, '?': 2, 'S': 3, ' ': 4, 'H': 5}
# return two lists of amino acid / label strings
def parse_file(data_file):
    X_strings = ''
    Y_strings = ''
    next_line_is_seq = False
    next_line_is_output = False
    # String formatting for the input 
    with open(data_file, 'r') as f:
        for line in f:
            if 'A:sequence' in line:
                next_line_is_output = False
                next_line_is_seq = True
                X_strings = X_strings + '<'
            elif 'A:secstr' in line:
                next_line_is_seq = False
                next_line_is_output = True
                Y_strings = Y_strings + '<'
            elif (':sequence' in line) or ('secstr' in line):
                next_line_is_seq = False
                next_line_is_output = False
            elif next_line_is_seq: 
                X_strings = X_strings + line[:-1]
            elif next_line_is_output:
                Y_strings = Y_strings + line[:-1]
    splitX = X_strings.split('<')[1:] # Input primary protein structure
    splitY = Y_strings.split('<')[1:] # Label secondary protein structure
    assert len(splitX) == len(splitY)
    return splitX, splitY

def create_sequence(string, vocab_dict):
    sequence = []
    # One hot encoding
    vec_length = np.max(list(vocab_dict.values()))
#     print vec_length
    for i in range(len(string)):
        temp = np.zeros(vec_length)
        temp[vocab_dict[string[i]]-1] = 1
        sequence.append(temp)
    return sequence

    
def pad_sequences(input_data, window_size=5):
    padded_sequence = []
    for i in tqdm.tqdm(range(len(input_data))):
        padded_sequence.append(np.pad(input_data[i], math.ceil(window_size/2), 'constant'))
    return padded_sequence

def create_window_data(input_data, output_data, window_size=21):
    windowed_input = []
    output = []
    # Padding input sequences on both sides with zeros
    input_data = pad_sequences(input_data=input_data, window_size=window_size)
    for j in tqdm.tqdm(range(len(output_data))):
        for i in range(len(output_data[j])):
            slice_ = input_data[j][i:i+window_size]
            windowed_input.append(np.array(slice_).astype(float))
            output.append(output_data[j][i])
    return np.array(windowed_input), output

def build_mlp(class_weight, train_data,input_data,output_data,model, num_nodes = 100, num_layers = 2, 
              activation='relu', ouput_activation='softmax',
              kernel_regularizer = None,batch_size = 100,optimizer = "adam"):
    """
    Builds a basic neural network in Keras.
    By default, assumes a multiclass classification 
    problem (softmax output).
    """
    
    
    model.add(Dense(num_nodes, activation=activation, 
                    input_shape=(train_data.shape[1],train_data.shape[2]),kernel_regularizer = None))
    model.add(Flatten())
    for l in range(num_layers-1):
        model.add(Dense(num_nodes, activation=activation,kernel_regularizer = regularizers.l2(0.01)))
        model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy']) 
    history = model.fit(input_data, output_data, validation_split = 0.2, epochs =2, batch_size = batch_size,class_weight = class_weight)
    return model,history

def jNN_msa(X):
    
    path = os.getcwd()
    path = os.path.join(path,"DATA")
    path_1 = os.chdir(path)
    
    X_strings,Y_strings,msa,file_lst = MSA_parser.parse_file(path_1)
    Y_strings = ThreeStateGenerator.replace_all(Y_strings)
    profile_table,X_strings = MSA_parser.making_profile_table(msa)
    input_sequences = []
    output_sequences = []
    for i in tqdm.tqdm(range(len(X_strings))):
        input_sequences.append(create_sequence(X_strings[i], ACIDS))
        output_sequences.append(create_sequence(Y_strings[i], LABELS))
            
    
    input_data, output_data = create_window_data(input_data=input_sequences, output_data=output_sequences)
    output_data = np.array(output_data)
     
    train_data = input_data[:int(0.8*input_data.shape[0])]
    train_labels = output_data[:int(0.8*output_data.shape[0])]
    test_data = input_data[int(0.8*input_data.shape[0]):]
    test_labels = output_data[int(0.8*output_data.shape[0]):]
    

    # To remove class imbalance....
    class_weight1 = class_weight.compute_class_weight('balanced', np.unique([np.argmax(i) for i in list(train_labels)]), [np.argmax(i) for i in list(train_labels)])
    
    model = Sequential()
    model, history = build_mlp(class_weight1, train_data,input_data,output_data,model, num_nodes=180, num_layers=2, 
                        activation='relu',kernel_regularizer = None,batch_size = 100,optimizer = "adam")
    
    ###Prediction
    Xnew,s = MSA_parser.making_profile_table(X)

    Ynew = '>' * len(s[0])
    lst = []
    lst.append(Ynew)
    
    
    
    input_sequences = []
    output_sequences = []
    for i in tqdm.tqdm(range(len(s))):
        input_sequences.append(create_sequence(s[i], ACIDS))
        output_sequences.append(create_sequence(lst[i], LABELS))
            
    
    input_data, output_data = create_window_data(input_data=input_sequences, output_data=output_sequences)
    
    output_data = np.array(output_data)
    
    output = model.predict(input_data)
    
        
    INV_LABELS = {}
    for key in LABELS.keys():
        INV_LABELS[LABELS[key]] = key
    prediction = ''.join([INV_LABELS[np.argmax(output[i])+1] for i in range(output.shape[0])])

    #pred = list(prediction)
    return prediction

if __name__=="__main__":


    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/code/DATA')
    #path = os.chdir('/home/noor/thesis/DATA')
    
    X_strings,Y_strings,msa,file_lst = MSA_parser.parse_file(path)

    
    
    ## Converts 8 states to 3 states -----------------CHECKED-----------------------
    Y_strings = ThreeStateGenerator.replace_all(Y_strings)
    
    profile_table,X_strings = MSA_parser.making_profile_table(msa)

             
    
    input_sequences = []
    output_sequences = []
    for i in tqdm.tqdm(range(len(X_strings))):
        input_sequences.append(create_sequence(X_strings[i], ACIDS))
        output_sequences.append(create_sequence(Y_strings[i], LABELS))
            
    
    input_data, output_data = create_window_data(input_data=input_sequences, output_data=output_sequences)
    
    output_data = np.array(output_data)
     
    train_data = input_data[:int(0.8*input_data.shape[0])]
    train_labels = output_data[:int(0.8*output_data.shape[0])]
    test_data = input_data[int(0.8*input_data.shape[0]):]
    test_labels = output_data[int(0.8*output_data.shape[0]):]
    

    # To remove class imbalance....
    #class_weight = class_weight.compute_class_weight('balanced', np.unique([np.argmax(i) for i in list(train_labels)]), [np.argmax(i) for i in list(train_labels)])
    

    #rmsprop = Adam(lr=learning_rate)
    #sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0, nesterov=False)
    """
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', 
                  optimizer="sgd",
                  metrics=['accuracy']) 
    """
    """
    history = model.fit(train_data, train_labels, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        class_weight = class_weight,
                        validation_data=(test_data, test_labels))
    """
    
    #history = model.fit(input_data, output_data, validation_split = 0.2, epochs = 70, batch_size = 100,class_weight = class_weight)
    
    
    
    model = Sequential()
    history = build_mlp(input_data,output_data,model, num_nodes=180, num_layers=2, 
                        activation='relu',kernel_regularizer = None,batch_size = 100,optimizer = "adam")
    
    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/tool')
    
    X = parse.parse_msa_file(path)
    Xnew,s = MSA_parser.making_profile_table(X)

    Ynew = '>' * len(s[0])
    fuck = []
    fuck.append(Ynew)
    
    
    
    input_sequences = []
    output_sequences = []
    for i in tqdm.tqdm(range(len(s))):
        input_sequences.append(create_sequence(s[i], ACIDS))
        output_sequences.append(create_sequence(fuck[i], LABELS))
            
    
    input_data, output_data = create_window_data(input_data=input_sequences, output_data=output_sequences)
    
    output_data = np.array(output_data)
    
    output = model.predict(input_data)
    
        
    INV_LABELS = {}
    for key in LABELS.keys():
        INV_LABELS[LABELS[key]] = key
    prediction = ''.join([INV_LABELS[np.argmax(output[i])+1] for i in range(output.shape[0])])
    print(prediction)
    print(len(prediction))
    pred = list(prediction)
    
    
    """
    print("--------------------")
    print("First 5 samples validation:", history.history["val_acc"][0:5])
    print("First 5 samples training:", history.history["acc"][0:5])
    print("--------------------")
    print("Last 5 samples validation:", history.history["val_acc"][-5:])
    print("Last 5 samples training:", history.history["acc"][-5:])
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    """
    
