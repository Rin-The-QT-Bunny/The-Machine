# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:26:37 2022

@author: doudou
"""

import torch # torch
import torch.nn as nn # neural networks
import torch.nn.functional as F # functionals 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # plot graph and figures

from Boomsday.ImageDSL import * # Image DSL operators
from PIL import Image # load images 
from karazhan.uruloki.parser_trial import * # decomposition of program tree

from config import *
from image_net import *
from DSL import *

class DSL_Base:
    def __init__(self):
        self.size = 0
        self.operators = ["root","whiteBoard","drawCircle","drawRectangle","Count","Union","Intersection","X","Y","Int256"]
        self.ops = [1,2,3,4,5,6,7,8,9] # Notice that root is not considered in this place
        self.arged_ops = ["Count","Union","Intersection","drawCircle","drawRectangle"] # these are operators with arguments
        self.arguments = [0,1,2,3,4,5] # notice that arg 0 is the root argumnet
        self.arg_dict = {"Count":[1],"Union":[2,3],"Intersection":[4,5],"drawCircle":[5,6,7,8],"drawRectangle":[9,10,11,12]}   # argument dictionary from operator to arguments
        
    def register_operator(self,op):
        return 0

# a patch fo the problem of zero parse
def FilterEmpty(List):
    returnList = []
    for i in range(len(List)):
        if (List[i] != ""):
            returnList.append(List[i])
    return returnList

class stochField(nn.Module):
    def __init__(self,s_dim = 64,dtype = "int256"):
        super().__init__()
        self.s_dim = s_dim
        self.fc1 = nn.Linear(s_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,2) # softplus and it will be done
        
    def parameterize(self,semantics):
        h = self.fc1(semantics)
        h = self.fc2(F.sigmoid(h))
        h = F.softplus(self.fc3(F.tanh(h)))
        return h
        
    def probLocalize(self,semantics,value):
        paras = self.parameterize(semantics)
        mu = paras[0][0]
        sigma = paras[0][1]
        return torch.exp(- (value-mu)*(value-mu)/(2 * sigma* sigma))
    def sample(self,semantics):
        paras = self.parameterize(semantics)
        mu = paras[0][0]
        sigma = paras[0][1]
        return mu
    
class VisualParser(nn.Module):
    # weaver of angband. use the encode information to weave the program understanding.
    
    def __init__(self,op_dim = 232, args_dim = 64,state_dim = 64,vocab = 3000):
        super(VisualParser, self).__init__()
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
        self.pfc1 = nn.Linear(op_dim+state_dim,300)
        self.pfc2 = nn.Linear(300,200)
        self.pfca = nn.Linear(200,200)
        self.pfc3 = nn.Linear(200,1)
        # to create the repeater unit
        self.rfc1 = nn.Linear(args_dim+state_dim,200)
        self.rfc2 = nn.Linear(200,399)
        self.rfc3 = nn.Linear(399,state_dim)
        # Gated recurrential unit
        self.fGRU = torch.nn.GRU(word_dim,state_dim,1)
        self.bGRU = torch.nn.GRU(state_dim,state_dim,1)
        self.namo = "Conceptualizer"
        self.DSL = DSL_Base()
        
        self.tsi = nn.Linear(args_dim,state_dim)
        
        self.sf  = stochField()
    
    def pdf_ops(self,opp,state):
        # shape of input variables are [num, embed_dim]
        # output will be the pdf tensor and the argmax position
        states = torch.broadcast_to(state,[len(opp),self.state_dim])

        r = torch.cat([opp,states],1)

        pdf_tensor = F.tanh(self.pfc1(r))
        pdf_tensor = F.softplus(self.pfc2(pdf_tensor))
        pdf_tensor = F.softplus(self.pfc3(pdf_tensor))
        #create pdf tensor
        pdf = pdf_tensor/torch.sum(pdf_tensor,0)
        
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
    
    def convert(self,x):
        # Start the parsing process
        self.log_p = 0
        self.count = 0
        self.program = ""
        def parse(s,arg,ops,ops_dict):
            # analogue goal-based policy in multi-goal RL
            # estimate the action to take when given the state (s) and the goal (arg)
            parse_state = self.repeat_vector(s,arg)
            # 1. create a state given the current state and the goal given
            pdf,index = self.pdf_ops(ops,parse_state)

            self.log_p  = self.log_p - torch.log(pdf[index])
            if (index == 8):
                operator = float(self.sf.sample(self.tsi(arg)+parse_state))
            else:
                print(ops_dict)
                operator = ops_dict[index+1]
            
                
            self.program += str(operator)
            if (str(operator) == "whiteBoard"):
                self.program += "()"
            # 2. pass the semantics down to next level if required
            self.count +=1
            
            if operator in self.DSL.arged_ops:
                self.program +=  "("
                
                args = self.DSL.arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    if (self.count < 5):
                        parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict)
                    else:
                        self.program += "RandOp"
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(self.DSL.ops))
        
        # TODO: dynamic library useage during the parsing
        parse(x,root,ops,self.DSL.operators)
        return self.program,self.log_p
    
    def program_prob(self,x,ops_sequence):
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
            try:
                op = float(ops_sequence[self.count])
                index = 9
            except:

                index = ops_dict.index(ops_sequence[self.count])

            self.log_p  = self.log_p - torch.log(pdf[index-1])
            operator = ops_sequence[self.count]
            try:
                op = float(ops_sequence[self.count])
                #print("Int256")
                self.log_p -= torch.log(self.sf.probLocalize(self.tsi(arg)+parse_state, op))

            except:
                pass

            self.program += str(operator)
            # 2. pass the semantics down to next level if required
            self.count +=1
            
            if operator in self.DSL.arged_ops:
                self.program +=  "("
                
                args = self.DSL.arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict,ops_sequence)
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(self.DSL.ops))
        
        # TODO: dynamic library useage during the parsing
        parse(x,root,ops,self.DSL.operators,ops_sequence)
        return self.program,self.log_p
    
    def save_model(self):
        dirc = "D:/SoulForge/" + str(self.namo) + "/"
        torch.save(self, dirc+self.namo + '.pth')
        
    def load_model(self):
        dirc = "D:/SoulForge/" + str(self.namo) + "/"
        try:
            self = torch.load(dirc+self.namo + '.pth')
        except:
            print("Failed")
            
    def run(self,x,seq = None):
        sequence = self.wordvecs(torch.tensor(x))
        
        results,hidden  = self.fGRU(sequence)
        final_r,final_h = self.bGRU(hidden)

        
        hidden = torch.reshape(final_h,[final_h.shape[1],self.state_dim])
        semantics = torch.reshape(hidden[hidden.shape[0]-1],[1,self.state_dim])
        if seq != None:
            program,loss  = self.program_prob(semantics,seq)
        else:
            program,loss = self.convert(semantics)
        return program, loss

class Conceptualizer(nn.Module):
    def __init__(self,function = None,w = 320,h = 480):
        super(Conceptualizer, self).__init__()
        self.function = function
        self.semanticsEncoder = EncoderNet(w,h,3)
        self.weaver = VisualParser()
    
    def forward(self,x):
        semantics = self.semanticsEncoder(x)
        program,treeRepresentation,confidence = None
        
    def train(self,function):
        proposal = exec(function)
        semantics = self.semanticsEncoder(proposal)
        p = self.weaver()
        
    def recognize(self,img):
        return self.semanticsEncoder(img)
    
    def trainPair(self,image,program):
        semanticsVector = self.recognize(image)
        seq = FilterEmpty(Decompose(program))
        program,loss = self.weaver.program_prob(semanticsVector,seq)
        return program, loss
    def representation(self,image):
        return self.weaver.convert(self.recognize(image))
        
    
class Deconstructor(nn.Module):
    def __init__(self):
        super(Deconstructor,self).__init__()
        self.RecurrentAttention = MultiAttention(320,480,5)
        self.Concept = Conceptualizer()
        self.Structures = []
        
    def destruct(self,image):
        self.Sturctures = []
        masks = self.RecurrentAttention(image)
        for i in range(len(masks)):
            maskImg = masks[i] * image
            code = self.Concept.representation(maskImg)
            self.Structures.append(code)
        
        return self.Structures