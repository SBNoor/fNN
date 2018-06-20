#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import glob,os
import ThreeStateGenerator
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import math
from sklearn.utils import class_weight
import MSA_parser

ACIDS = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N':11, 'Q':12, 'P':13, 'S':14, 'R':15, 'T':16, 'W':17, 'V':18, 'Y':19, 'X':20, 'U': 21, 'Z': 22, 'B':23, 'O':24,'.':25}
LABELS = {' ': 1, 'H': 2, 'S': 3}

def parse_file(path):
    
    X_str = []
    Y_str = []
    sequence = []
    file_lst = []
    
    
    for data_file in glob.glob("*.all"):
        file_lst.append(data_file)
        with open(data_file, 'r') as f:
            alignments = []
            content = f.readlines()
            for line in content:
                if "RES" in line:
                    seq_obt = line.split(":")
                    seq = seq_obt[1].rstrip().split(",")
                    X_str.append(''.join(seq))
                if "DSSP" in line and not "DSSPACC" in line:
                    sec_seq_obt = line.split(":")
                    sec_seq = sec_seq_obt[1].rstrip().split(",")
                    Y_str.append(''.join(sec_seq))
                if "align" in line:
                    align_seq_obt = line.split(":")
                    align_seq = align_seq_obt[1].rstrip().split(",")
                    
                    alignments.append(''.join(align_seq))
                    
        sequence.append(alignments)
                    
    return X_str,Y_str,sequence,file_lst

def filtering_sequences(sequence, file_lst):
    files = []
    for j in range(len(sequence)):
        counter = 0
        amino_acid = [list(character) for character in sequence[j]]
        transposed_amino_acid = list(map(list, zip(*amino_acid)))
        flag_1 = True
        flag_2 = False
        for i in range(len(transposed_amino_acid)):
    
            flag_1 = any('.' in word for word in transposed_amino_acid[i])
    
            flag_2 =  transposed_amino_acid[i][1:] == transposed_amino_acid[i][:-1]
    
            if (flag_1 == False) and (flag_2 == True):
                counter += 1
            percentage = counter / len(sequence[j][0])
            flag_1 = True
            flag_2 = False
                
        if percentage < 100:
            files.append(file_lst[j])

        counter = 0
                
    return files

def filtering_y_strings(Y_string,files,file_lst):
    Y_strings = []
    for j in range(len(file_lst)):
        if any(file_lst[j] in s for s in files):
            Y_strings.append(Y_string[j])
    return Y_strings
        
    
def pad_profile_table(transposed_table,window_size):
    padded_table = np.pad(transposed_table, ((0,0),(math.floor(window_size/2),math.floor(window_size/2))), mode='constant', constant_values=0)
    
    return padded_table

def making_profile_table(msa, file_lst,files,window_size):
    
    seq_lst = []
    collapsed = []
    counter = 0
    for i in range(len(msa)):
        if any(file_lst[i] in s for s in files):
            counter += 1
            amino_acid = [list(character) for character in msa[i]]
            transposed_amino_acid = list(map(list, zip(*amino_acid)))
            profile_table = np.zeros(shape = (len(transposed_amino_acid),26),dtype = int)
            for j in range(len(transposed_amino_acid)):
                for k in range(len(transposed_amino_acid[j])):
                    if transposed_amino_acid[j][k] != '.':
                        index = ACIDS[transposed_amino_acid[j][k]]
                        profile_table[j][index] = profile_table[j][index] + 1
                        
            padded_table = pad_profile_table(np.transpose(profile_table),window_size)
            collapsed.append(create_window_data(padded_table,window_size,True))

            
                    
    return profile_table,seq_lst, collapsed, padded_table

def create_window_data(padded_table,window_size,flag):
    window_data = []
    j = window_size
    
    for i in range(len(padded_table[0]) - (window_size-1)):
        extracted_table = padded_table[:,i:j]
        j += 1
        if flag == True:
            window_data.append((np.sum(extracted_table, axis = 1))/np.sum(padded_table)) 
        else:
            window_data.append(extracted_table)
    
    return window_data 

def data_shaping(sums, Y_strings):
    
    input_data = []
    for i in range(len(sums)):
        for j in range(len(sums[i])):
            input_data.append(sums[i][j])
        
    input_data = np.array(input_data)

    output_data = []
    for i in range(len(Y_strings)):
        for j in range(len(Y_strings[i])):
            output_data.append(Y_strings[i][j])

    output_data = np.array(output_data)
    
    return input_data, output_data

def create_window_data_for_second_network(output,window_size):
    
    window = []
    table = pad_profile_table(np.transpose(output),window_size)
    
    window.append(create_window_data(table,window_size,False))
    
    return window,table

def obtaining_sequence(output,dictionary):
    seq = []
    for i in range(len(output)):
            pos = np.argmax(output[i],axis = 0)
            structure = [key for key, value in dictionary.items() if value == pos][0]
            seq.append(structure)

    return seq

def the_nn1(X, Y, node1, batch, epoch, optimizer, reg, class_weight, topred):
    model = Sequential()
    model.add(Dense(node1, input_dim = 26, activation = "relu",
                    kernel_regularizer=reg))
    model.add(Dropout(0.2))
    model.add(Dense(node1, input_dim = 26, activation = "relu",
                    kernel_regularizer=reg))
    model.add(Dropout(0.2))
    model.add(Dense(len(Y[1]), activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["acc"])

    history = model.fit(X, Y, validation_split = 0.2, epochs = epoch, batch_size = batch, 
                        class_weight = class_weight)
    
    output_topass = model.predict(X)
    output_topred = model.predict(topred)
    
    return history, model, output_topass, output_topred
    
def the_nn2(X2, Y2, node2, batch, epoch, optimizer, reg, class_weight):
    model = Sequential()
    model.add(Dense(node2, input_shape = (3, len(X2[0][2])), activation = "relu",
                    kernel_regularizer=reg))
    model.add(Flatten(input_shape=X2.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(node2, activation = "relu",
                    kernel_regularizer=reg))
    model.add(Dropout(0.2))
    model.add(Dense(len(Y2[1]), activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["acc"])

    history = model.fit(X2, Y2, validation_split = 0.2, epochs = epoch, batch_size = batch,
                        class_weight = class_weight)

    return history, model

def mNN(file):
    path = os.getcwd()
    msa1, seq_lst1 = MSA_parser.making_profile_table(file)
    
    path = os.path.join(path,"DATA")
    path_1 = os.chdir(path)
        
    seq_lst,Y_strings,msa,file_lst = parse_file(path_1)
    files = filtering_sequences(msa,file_lst)   
    Y_strings = ThreeStateGenerator.replace_all(Y_strings)
    Y_strings = filtering_y_strings(Y_strings,files,file_lst) 
    profile_table,X_strings,sums,padded_table = making_profile_table(msa, file_lst,files,
                                                                     window_size = 7)
    input_data, output_data = data_shaping(sums, Y_strings)
    # integer encoding of the labels
    encY = LabelEncoder()
    encY.fit(output_data)
    encoded_Y = encY.transform(output_data)
    cw = class_weight.compute_class_weight('balanced', np.unique(encoded_Y), encoded_Y)
    # one hot encoding
    onehot_Y = np_utils.to_categorical(encoded_Y)
    X = input_data
    Y = np.asarray(onehot_Y)   
    
    ###### TEST DATA
    path = os.chdir("..")
    padded_table1 = pad_profile_table(np.transpose(msa1), 7)
    
    sums1 = []
    
    sums1.append(create_window_data(padded_table1,7,True))
    
    input_data1, output_data1 = data_shaping(sums1, Y_strings)

    X1 = input_data1
    history, model, output_topass, output_topred = the_nn1(X, Y, 10, 200, 200, "adam", None, cw, X1)
 
    ####### TRAINING DATA    
    second_window_data,table = create_window_data_for_second_network(output_topass,7)
    input_second = np.asarray(second_window_data[0])

    ####### TEST DATA    
    second_window_data1,table1 = create_window_data_for_second_network(output_topred,7)
    input_second1 = np.asarray(second_window_data1[0])

    to_predict = input_second1
    
    history2, model2 = the_nn2(input_second, Y, 100, 200, 200, "adam", None, cw)
    output = model2.predict(to_predict)
    
    INV_LABELS = {}
    for key in LABELS.keys():
        INV_LABELS[LABELS[key]] = key
    prediction = ''.join([INV_LABELS[np.argmax(output[i])+1] for i in range(output.shape[0])])
    print("prediction : ",prediction)
    
    return prediction

