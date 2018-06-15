#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import glob,os
import numpy as np

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

def reading_pssm_files(content):
    pssm = []
    file_lst = []

    #with open(data_file, 'r') as f:
    #content = f.readlines()
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
    
    file_lst.append('user_input')
            
    assert len(pssm) == len(file_lst)
    
    return pssm
            

if __name__=="__main__":
    #path = os.getcwd()
    #path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/code/DATA')
    #X_strings, Y_strings, file = parse_file(path)
    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/tool')
    X = parse_msa_file(path)
    