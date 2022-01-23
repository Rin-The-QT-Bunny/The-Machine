# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:20:13 2021

@author: doudou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import time # record the time record
import matplotlib.pyplot as plt #draw the diagram

import numpy as np #linear algebra
import datetime
import random
from DSL import *

from math import *
from Boomsday.omegas.dataloader import *
from config import *
import config as config

data = Mathloader(path,trainNum)

symbol_id_collection = []
# gain symbol id from tokenizer
for i in range(10):
    st = "the value symbol" + str(i) + " and what is that "
    idx = data.tokenize_data(st)[0][2]
    symbol_id_collection.append(idx)


def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

class SymbolNet(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.id = name
        self.wordvecs = nn.Embedding(10000, 232)
    
    def forward(self,x):
        # shape : [1,n]
        
        symbolCount = 0
        # 1.step:
        X = torch.tensor(x)
        wvectors = self.wordvecs(X)
        embeds =wvectors
        reserves = []
        # 2.step:
        for i in range(len(x[0])):
            item = x[0][i]
            word = data.tokenizer_en.sequences_to_texts([[item]])
            if ("symbol" in word[0]):
                reserves.append(i)
        
        # 3.step:
        symbols = embeds[0][idx]
        symbols = torch.reshape(symbols,[1,-1])
        for i in range(len(reserves)-1):
            symbols = torch.cat([symbols,torch.reshape(embeds[0][i+1],[1,-1])],0)

        return torch.tensor(symbols)
        



class Parser(nn.Module):
    def __init__(self,op_dim = 232, args_dim = 64,state_dim = 256,word_dim = 232, vocab = 10000):
        super(Parser, self).__init__()
        
        # create a list of vectors representing operators and words
        self.wordvecs = nn.Embedding(vocab, word_dim)
        self.op_vecs = nn.Embedding(vocab, op_dim)
        self.arg_vecs = nn.Embedding(vocab, args_dim)

        # Input of this model will be a set of embeddings of the operator
        
        # Recreate operators using pytorch
        self.state_dim = state_dim
        self.args_dim = args_dim
        self.op_dim = op_dim
        self.word_dim = word_dim
        
        # Synthesis methods required
        self.log_p = 0
        self.count = 0
        self.program = ""
        
        # to create the probability analyzer unit
        self.pfc1 = nn.Linear(state_dim,100)
        self.pfc2 = nn.Linear(100,64)
        #self.pfca = nn.Linear(64,100)
        self.pfc3 = nn.Linear(64,word_dim)
        # to create the repeater unit
        self.rfc1 = nn.Linear(args_dim+state_dim,200)
        self.rfc2 = nn.Linear(200,100)
        self.rfc3 = nn.Linear(100,state_dim)
        # Gated recurrential unit
        self.fGRU = torch.nn.GRU(word_dim,state_dim,1)
        self.bGRU = torch.nn.GRU(state_dim,state_dim,1)
        self.namo = "Math-Reaver"
        
        self.words = None
        self.attention_data = []
        self.DSL =MathDSL
        
        self.STCM = []
        self.sn = SymbolNet("LilRag")
    
    def attention(self,q,words):
        att = torch.softmax(torch.matmul(words,q.permute(1,0)),-2)
        self.attention_data.append(att)
        return torch.matmul(att.permute(1,0),words)
    
    def pdf_ops(self,opp,order_state):
        # shape of input variables are [num, embed_dim]
        # output will be the pdf tensor and the argmax position
        
        query = F.tanh(self.pfc1(order_state))
        query = F.softplus(self.pfc2(query))
        query = F.softplus(self.pfc3(query))
        
        fused = self.attention(query,self.words)


        #create pdf tensor

        pdf = torch.matmul(fused,opp.permute(1,0))

        pdf = torch.softmax(pdf,-1)[0]
        
        #index of the maximum operator is
        index = np.argmax(pdf.detach().numpy())
        return pdf, index
    
    def repeat_vector(self,semantics,arg):
        # Create next semantics vector for the argument
        r = torch.cat([semantics,arg],-1)
        r = F.tanh(self.rfc1(r))
        r = F.softplus(self.rfc2(r))
        r = F.tanh(self.rfc3(r))
        
        return r
    
    def convert(self,x,words):
        self.attention_data = []
        self.words = words
        # Start the parsing process
        self.log_p = 0
        self.program = ""
        self.count = 0
        def parse(s,arg,ops,ops_dict):
            # analogue goal-based policy in multi-goal RL
            # estimate the action to take when given the state (s) and the goal (arg)
            parse_state = self.repeat_vector(s,arg)
            # 1. create a state given the current state and the goal given
            pdf,index = self.pdf_ops(ops,parse_state)
            #index = ops_dict.index(ops_sequence[self.count])
            
            self.log_p  = self.log_p - torch.log(pdf[index])
            
            operator = ops_dict[index+1]
            self.program += str(operator)
            self.count += 1
            #print(ops_dict,index)
            # 2. pass the semantics down to next level if required

            if operator in self.DSL.arged_ops:
                self.program +=  "("
                
                args = self.DSL.arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    if self.count < 35:
                        parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict)
                    else:
                        self.program+="RandO"
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(self.DSL.ops))
        ops= torch.cat([ops,self.STCM],0)
        
        # TODO: dynamic library useage during the parsing
        operators = []
        operators.extend(self.DSL.operators)
        operators.extend(self.STOP)
        # TODO: dynamic library useage during the parsing
        parse(x,root,ops,operators)
        return self.program,self.log_p
    
    def program_prob(self,x,ops_sequence,words):
        self.attention_data = []
        self.words = words
        # Start the parsing process
        self.log_p = 0
        self.count = 0
        self.program = ""
        
        def parse(s,arg,ops,ops_dict,ops_sequence):
            # analogue goal-based policy in multi-goal RL
            # estimate the action to take when given the state (s) and the goal (arg)
            parse_state = self.repeat_vector(s,arg)
            # 1. create a state given the current state and the goal given
            
            pdf,index = self.pdf_ops(ops,parse_state)
            index = ops_dict.index(ops_sequence[self.count])

            self.log_p  = self.log_p - torch.log(pdf[index-1])
            operator = ops_dict[index]

            self.program += str(operator)

            # 2. pass the semantics down to next level if required
            self.count += 1
            
            if operator in self.DSL.arged_ops:
                self.program +=  "("
                
                args = self.DSL.arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    if self.count < 35:
                        parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict,ops_sequence)
                    else:
                        self.program+="RandO"
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(self.DSL.ops))
        ops= torch.cat([ops,self.STCM],0)
        
        # TODO: dynamic library useage during the parsing
        operators = []
        operators.extend(self.DSL.operators)
        operators.extend(self.STOP)

        parse(x,root,ops,operators,ops_sequence)
        return self.program,self.log_p
    
    def save_model(self):
        dirc = "D:/Omegas/" + str(self.namo) + "/"
        torch.save(self, dirc+self.namo + '.pth')
        
    def load_model(self):
        dirc = "D:/Omegas/" + str(self.namo) + "/"
        try:
            self = torch.load(dirc+self.namo + '.pth')
        except:
            print("Failed")
            
    def run(self,x,symbols,seq = None):
        self.STCM = self.sn(x)
        self.STOP = symbols
        sequence = self.wordvecs(torch.tensor(x))
        
        results,hidden  = self.fGRU(sequence)
        final_r,final_h = self.bGRU(hidden)

        
        hidden = torch.reshape(final_h,[final_h.shape[1],self.state_dim])
        semantics = torch.reshape(hidden[hidden.shape[0]-1],[1,self.state_dim])
        if seq != None:
            program,loss  = self.program_prob(semantics,seq,sequence[0])
        else:
            program,loss = self.convert(semantics,sequence[0])
        return program, loss
    
    def train(self,tasks,EPOCH):
       optimizer =torch. optim.Adam(psr.parameters(), lr=2e-4)
       for epoch in range(EPOCH):
           Loss = 0.0
           for i in range(len(tasks)):
               seq = tasks[i][0]
               #print(tasks[i][1])
               #print(tasks[i][2])
               print(data.tokenizer_en.sequences_to_texts(tasks[i][0]))
            
               pro,loss = psr.run(seq,tasks[i][3],tasks[i][1])
               print("Program:",pro)
               print("Loss:",loss.detach().numpy())
               Loss += loss/10
               print(" ")
           optimizer.zero_grad()
           Loss.backward()
           optimizer.step()
           if epoch%100 == 0:
               print("EPOCH:",epoch,"Loss is ",Loss.detach().numpy())
    
def visualize_masks(masks):
    try:
        img = torch.cat([masks[0],masks[1]],-1)

        for i in range(len(masks)-2):
            img = torch.cat([img,masks[i+2]],-1)
    except:
        img = masks[0]
    img = img.permute(1,0)
    plt.cla()
    plt.imshow((img.view(1,masks.shape[0])).detach().numpy(),'bone')
    plt.pause(0.01)
       
def visualize_attention(att):
    plt.cla()
    try:
        img = torch.cat([att[0],att[1]],-1)
        for i in range(len(att)-2):
            img = torch.cat([img,att[i+2]],-1)
    except:
        img = att[0]
        
    plt.imshow(img.detach().numpy())
    plt.pause(1)
        
psr = Parser()
tasks = data.train_set[:10]
from karazhan.curator import *
tasks = data.make_tasks(ArithMatics)
psr.train(tasks,3000)


psr.run(tasks[0][0],tasks[0][3])
