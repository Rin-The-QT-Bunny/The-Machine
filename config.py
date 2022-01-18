# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:21:03 2022

@author: doudou
"""
# setup basic configuration
word_dim =  1 # the representation dimension of words
state_dim = 128 # the representation dimension of semantics
ops_dim = 128 # represenatation dimension of operators
arg_dim = 64 # representation dimension of arguments

max_len = 30 # The maximum length of a sentence
EPOCH = 2400  # training epoch shown here

beamSize = 4 # the size of the task frontier
itersAttenntion = 3 # the number of objects to decompose