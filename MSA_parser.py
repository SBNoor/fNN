#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:33:00 2018

@author: Noor
"""

import glob,os
import ThreeStateGenerator
import numpy as np

ACIDS = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N':11, 'Q':12, 'P':13, 'S':14, 'R':15, 'T':16, 'W':17, 'V':18, 'Y':19, 'X':20, 'U': 21, 'Z': 22, 'B':23, 'O':24,'.':25}


def parse_file(path):
    
    X_str = []
    Y_str = []
    sequence = []
    file_lst = []
    
    
    for data_file in glob.glob("*.all"):
        file_lst.append(data_file)
        #print("data file : ",data_file)
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
                    #Y_str.append(sec_seq)
                if "align" in line:
                    align_seq_obt = line.split(":")
                    align_seq = align_seq_obt[1].rstrip().split(",")
                    
                    alignments.append(''.join(align_seq))
                    #alignments.append(align_seq[:-1])
                    
        sequence.append(alignments)
                    
    return X_str,Y_str,sequence,file_lst


def making_profile_table(msa):
    
    seq_lst = []
    
    for i in range(len(msa)):
        amino_acid = [list(character) for character in msa[i]] ### checked
        #print("amino fucking acid : ",amino_acid)
        transposed_amino_acid = list(map(list, zip(*amino_acid)))
        #print("transposed fucking acid : ",transposed_amino_acid)
        profile_table = np.zeros(shape = (len(transposed_amino_acid),26),dtype = int)
        for j in range(len(transposed_amino_acid)):
            for k in range(len(transposed_amino_acid[j])):
                if transposed_amino_acid[j][k] != '.':
                    #print("first char : ",transposed_amino_acid[j][0])
                    #if transposed_amino_acid[j][k] != '-':
                    index = ACIDS[transposed_amino_acid[j][k]]
                    profile_table[j][index] = profile_table[j][index] + 1
                
        seq_lst = majority_vote(profile_table,seq_lst)
                    
    return profile_table,seq_lst


def majority_vote(profile_table,seq_lst):
    seq = ''
    
    counter = 0
    for i in range(len(profile_table)):
        max_num = np.max(profile_table[i])
        lst = list(profile_table[i])
        max_index = lst.index(max_num)
        #if max_num != 0:
        aa = [k for k, v in ACIDS.items() if (v == max_index)][0]
        #print(counter + 1, "   aa : ",aa, "  max num : ",max_num)
            #print("max nym : ", max_num, "   aa : ",aa,"  ",end = "")
        counter += 1
        seq += aa
    seq_lst.append(seq)
    
    return seq_lst

#def returning_to_jens(seq_lst,Y_strings):
 #   return seq_lst,Y_strings
 
def check_length(seq_lst,X_strings,file_lst):
    for i in range(len(seq_lst)):
        if len(seq_lst[i]) != len(X_strings[i]):
            print("file  : ",file_lst[i])



if __name__=="__main__":
    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/code/DATA')
    #X_strings, Y_strings, file = parse.parse_file(path)
    seq_lst,Y_strings,msa,file_lst = parse_file(path)
    
    ## Converts 8 states to 3 states -----------------CHECKED-----------------------
    Y_strings = ThreeStateGenerator.replace_all(Y_strings)
    
    profile_table,X_strings = making_profile_table(msa)
    print("lebgth : ",len(profile_table[0]))
    print(seq_lst)
    print(X_strings)
    
    check_length(seq_lst,X_strings,file_lst)
    


    
    