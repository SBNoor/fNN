#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import glob, os
import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
import math
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd




ACIDS = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N':11, 'Q':12, 'P':13, 'S':14, 'R':15, 'T':16, 'W':17, 'V':18, 'Y':19, 'X':20, 'U': 21, 'Z': 22, 'B':23, 'O':24,'.':25}
LABELS = {'C': 1, 'E': 2, 'H': 3}

def reading_pssm_files(path_pssm):
    pssm = []
    file_lst = []
    for data_file in glob.glob("*.pssm"):
        #print(data_file)
        file_lst.append(data_file)
        with open(data_file, 'r') as f:
            content = f.readlines()
            content = content[3:len(content)-5]
            table = np.zeros(shape = (len(content)-1,20),dtype = int)
            result = [i.replace('\n','') for i in content]
            result = [i.lstrip() for i in result]
            result = result[:len(result)-1]

            counter = 0
            for i in range(len(result)):
                res = result[i].split(" ")
                res[:] = [item for item in res if item != '']
                res = res[2:22]
                for i in range(len(res)):
                    table[counter][i] = res[i]
                counter += 1
            
            
            pssm.append(table)
            
    assert len(pssm) == len(file_lst)
    
    return pssm,file_lst

def reading_predictions(path_predictions, filename):
    file_lst_predictions = []
    prediction = []
    with open(filename,'r') as f:
        content = f.readlines()
        content = [i.replace('\n','') for i in content]
        for line in content:
            if line[0] == '>':
                file_lst_predictions.append(line[1:])
            else:
                prediction.append(line)
    
    #assert len(file_lst_predictions) == len(prediction)
    
    return prediction,file_lst_predictions
        
def filtering_files(pssm,file_lst,Y_strings,file_lst_predictions):
    pssm_final = []
    file_lst_final = []
    for i in range(len(file_lst_predictions)):
        for j in range(len(file_lst)):
            if file_lst_predictions[i] in file_lst[j]:
                file_lst_final.append(file_lst[j])
                pssm_final.append(pssm[j])
    
    #assert len(pssm_final) == len(Y_strings)
    #assert len(file_lst_predictions) == len(file_lst_final)
    
    return np.transpose(np.vstack(pssm_final)),file_lst_final   

"""
wtf = []
for i in range(len(file_lst)):
    if "HUMAN" in file_lst[i]:
        wtf.append(file_lst[i])
"""       
def create_window_data(final_pssm, win_size):
    window_data = []
    j = win_size
    
    padded_table = np.pad(final_pssm, ((0,0),(math.floor(win_size/2),math.floor(win_size/2))), 
                          mode='constant', constant_values=0)

    for i in range(len(padded_table[0]) - (win_size-1)):
        extracted_table = padded_table[:,i:j]
        j += 1
        window_data.append(extracted_table)
    
    return np.array(window_data)

def getting_output(Y_strings):
    output_data = []
    for i in range(len(Y_strings)):
        for j in range(len(Y_strings[i])):
            output_data.append(Y_strings[i][j])
            
    encoder = LabelEncoder()
    encoder.fit(output_data)
    encoded_Y = encoder.transform(output_data)
    onehot_Y = np_utils.to_categorical(encoded_Y)
    
    dictionary = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return onehot_Y, dictionary


def the_nn(X, Y, val_split, epochs, batch_size, node1, node2, reg, opti, filter1, filter2):
    model = Sequential() 
    model.add(Conv2D(node1, filter1, input_shape=(X.shape[1], X.shape[2], 1), padding='same', activation='relu',
                     kernel_regularizer=reg))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Conv2D(node2, filter2, activation='relu', padding='same',
                     kernel_regularizer=reg))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

    history = model.fit(X, Y, validation_split = val_split, epochs = epochs, batch_size = batch_size)

    return model, history, output

def baseline_model():
    model = Sequential() 
    model.add(Conv2D(96, (5, 5), input_shape=(X.shape[1], X.shape[2], 1), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l1(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (2, 2), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l1(0.01)))
    model.add(Flatten())
    model.add(Dense(Y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    return model


    
def sNN(X):
    path = os.getcwd()
    path = os.path.join(path,"pssm")
    path_1 = os.chdir(path)
    
    ### TRAINING
    pssm,file_lst = reading_pssm_files(path)
    Y_strings,file_lst_predictions = reading_predictions(path, "psipred.txt")
    pssm_final,file_lst_final = filtering_files(pssm,file_lst,Y_strings,file_lst_predictions)
    windows = create_window_data(pssm_final, 21)
    Y, dictionary = getting_output(Y_strings)
    X = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
    
    Y_test,file_lst_test = reading_predictions(path_1, X)
    
    pssm_test = []
    file_lst_test2 = []
    for i in range(len(file_lst_test)):
        for j in range(len(file_lst)):
            if file_lst_test[i] in file_lst[j]:
                file_lst_test2.append(file_lst[j])
                pssm_test.append(pssm[j])
    
        pssm_test = np.transpose(np.vstack(pssm_test))
        windows_test = create_window_data(pssm_test, 21)
        Yt, dictionary2 = getting_output(Y_test)
        Xt = windows_test.reshape(windows_test.shape[0], windows_test.shape[1], windows_test.shape[2], 1)
    
    
        model, history = the_nn(X, Y, 0.2, 1, 1000, 96, 
                                    10, regularizers.l1(0.01), "adam", (5, 5), (2, 2), Xt)
    
    
        output= model.predict(Xt)
        
        INV_LABELS = {}
        for key in LABELS.keys():
            INV_LABELS[LABELS[key]] = key
        prediction = ''.join([INV_LABELS[np.argmax(output[i])+1] for i in range(output.shape[0])])
        print("prediction : ",prediction)
    
    return prediction