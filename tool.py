#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import argparse
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
import jnn
import os
import jsnn
import parse
import mnn
import snn



"""
def Main():
    
    #lst = ['--h','--j','--s','--js','--m']
    
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument('--file', type=argparse.FileType('a'),
                   help='Filename to write cuts to')
    parser.add_argument('-a', action="store_true", default=False)
    parser.add_argument('-b', action="store", dest="b")
    parser.add_argument('-c', action="store", dest="c", type=int)
    
    print('print_usage output:')
    parser.print_usage()
    print()
    
    print('print_help output:')
    parser.print_help()


    if parser.file:
        print("reading file")
"""
def check(content):
    flag = 0
    for i in range(len(content)):
        if content[i][0] == '>':
            flag += 1
    
    return flag

def main():
    
    file_name = []
    file = []
    flag = 0
    
    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()    
    parser.add_argument('file', type=argparse.FileType('r'))
    #parser.add_argument('file_msa', type=argparse.FileType('r'))
    parser.add_argument('--jNN','-j',help='Runs simple NN')
    parser.add_argument('--msa','-js',help = "Runs NN based on majority vote")
    parser.add_argument('--mNN','-m',help = "Runs cascaded NN")
    parser.add_argument('--sNN','-s',help = "Runs convolutional NN")
    parser.add_argument("--output", "-o", help="Output the result to a file", action="store_true")
    
    args = parser.parse_args()

    if args.file:           
        with args.file as f:
            content = f.readlines()
            flag = check(content)
            if flag == 1:
                
                a = ''
                for line in content[1:]:
                    line = line.strip('\n')
                    a += line
                file.append(a)
                file_name.append(content[0])
                
            
                print(file)
            elif flag > 1:
                file = parse.parse_msa_file(content)
                print(file)
            else:
                print("pssm")
        

    

    if args.jNN: 
        cm = jnn.jNN(file)
        c = cm.replace(" ", "C")
    
            
        if args.output:
            f = open("prediction.txt","a")
            f.write(">" + file_name[0]+  '\n')
            f.write(str(c) + '\n')
            f.close()
        else:
            print("Prediction : ")
            print((c))
        
    elif args.msa:
        cm = jsnn.jNN_msa(file)
        c = cm.replace(" ","C")
        
        if args.output:
            f = open("Prediction.txt","a")
            f.write(">prediction")
            f.write(str(c) + '\n')
            f.close()
        else:
            print("Prediction : ")
            print(c)
            
    elif args.mNN:
        cm = mnn.mNN(file)
        c = cm.replace(" ","C")
        
        if args.output:
            f = open("Prediction.txt","a")
            f.write(">prediction")
            f.write(str(c) + '\n')
            f.close()
        else:
            print("Prediction : ")
            print(c)
            
    elif args.sNN:
        cm = snn.sNN(file)
        c = cm.replace(" ","C")
        
        if args.output:
            f = open("Prediction.txt","a")
            f.write(">prediction")
            f.write(str(c) + '\n')
            f.close()
        else:
            print("Prediction : ")
            print(c)
    
    else:
        if flag == 1:
            cm = jnn.jNN(file)
            c = cm.replace(" ", "C")
        
                
            if args.output:
                f = open("prediction.txt","a")
                f.write(">" + file_name[0]+  '\n')
                f.write(str(c) + '\n')
                f.close()
            else:
                print("Prediction : ")
                print((c))
        elif flag > 1:
                    cm = mnn.jNN_msa(file)
        c = cm.replace(" ","C")
        
        if args.output:
            f = open("Prediction.txt","a")
            f.write(">prediction")
            f.write(str(c) + '\n')
            f.close()
        else:
            print("Prediction : ")
            print(c)
    
    
        
    
        
if __name__ == '__main__':
    print('\n')
    print('\n')
    print("##############################################################")
    print("3-state protein secondary structure prediction tool.")
    print("##############################################################")
    
    print('\n')
    print('\n')
    main()