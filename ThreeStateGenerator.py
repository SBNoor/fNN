#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:20:04 2018

@author: Noor
"""

import parse
import os
import string

INTAB = "GHIEBS_T"
OUTTAB= "HHHSS   "

def replace_all(Y_strings):
    Y_string = []
    
    for seq in Y_strings:
        Y_string.append(seq.translate(str.maketrans(INTAB, OUTTAB)))
    
    assert len(Y_strings) == len(Y_string)
    #print(Y_string[1])    
    return Y_string

if __name__=="__main__":
    path = os.chdir('/Volumes/Noor/Aarhus University/Semester 4/code/DATA')
    X_strings, Y_strings, file = parse.parse_file(path)
    Y_strings = replace_all(Y_strings)
    