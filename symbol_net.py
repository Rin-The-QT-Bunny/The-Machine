# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:51:17 2022

@author: doudou
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import *

symbol_id_collection = []
# gain symbol id from tokenizer
for i in range(10):
    st = "symbol" + str(i)
    idx = data.convert_to_data(st)[0][0]
    symbol_id_collection.append(idx)


class SymbolNet(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.id = name
        self.wordvecs = nn.Embedding(10000, 232)
    
    def forward(self,x):
        # shape : [1,n]
        symbols = {}
        symbolCount = 0
        # 1.step:
        wvectors = self.wordvecs(x)
        embeds = []
        # 2.step:
        for i in range(len(x[0])):
            item = x[0][i]
            if (item in symbol_id_collection):
                symbols["symbol"+str(symbolCount)] = embeds[i]
        # 3.step:
        return symbols
        
        
sn = SymbolNet("LilRag")
