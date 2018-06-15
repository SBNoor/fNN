#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import glob,os

def parse_file(path):
    splitX = []
    splitY = []
    file = []
    

    for data_file in glob.glob("*.all"):
        file.append(data_file)
        with open(data_file, 'r') as f:
            line1, line2 = next(f).rstrip(), next(f).strip()
            
            #Splitting the amino acid sequences
            seq_obt = line1.split(":")
            seq = seq_obt[1].split(",")
            splitX.append(''.join(seq))
            
            #Splitting the secondary structure
            sec_seq_obt = line2.split(":")
            sec_seq = sec_seq_obt[1].split(",")
            splitY.append(''.join(sec_seq))
            
    assert len(splitX) == len(splitY)
    return splitX,splitY,file

def parse_msa_file(content):
    string = ""
    for line in content:
        if line[0] == '>':
            string += '<'
        else:
            line = line.replace('-','.')
            string += line.rstrip()
    
    X = list(string.split('<'))
    #print(X)
                
    res = []
    res.append(X[1:])            
    return res
            

if __name__=="__main__":
    #path = os.getcwd()
    #path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/code/DATA')
    #X_strings, Y_strings, file = parse_file(path)
    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/tool')
    X = parse_msa_file(path)
    