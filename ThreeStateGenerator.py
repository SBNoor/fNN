#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Bakhtawar Noor and Judit Kisistók
Aarhus University, 2018
"""

import parse
import os
import string

INTAB = "GHIEBS_T?"
OUTTAB= "HHHSS    "

def replace_all(Y_strings):
    Y_string = []
    
    for seq in Y_strings:
        Y_string.append(seq.translate(str.maketrans(INTAB, OUTTAB)))
    
    assert len(Y_strings) == len(Y_string)
    return Y_string