#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
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


def making_profile_table(msa):
    
    seq_lst = []
    
    for i in range(len(msa)):
        amino_acid = [list(character) for character in msa[i]] 
        transposed_amino_acid = list(map(list, zip(*amino_acid)))
        profile_table = np.zeros(shape = (len(transposed_amino_acid),26),dtype = int)
        for j in range(len(transposed_amino_acid)):
            for k in range(len(transposed_amino_acid[j])):
                if transposed_amino_acid[j][k] != '.':
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
        aa = [k for k, v in ACIDS.items() if (v == max_index)][0]
        counter += 1
        seq += aa
    seq_lst.append(seq)
    
    return seq_lst

 
def check_length(seq_lst,X_strings,file_lst):
    for i in range(len(seq_lst)):
        if len(seq_lst[i]) != len(X_strings[i]):
            print("file  : ",file_lst[i])
    