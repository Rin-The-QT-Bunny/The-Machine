# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:17:17 2022

Title: Conceptualizer

@author: doudou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from Boomsday.ImageDSL import *
from PIL import Image
from karazhan.uruloki.parser_trial import *


path = "C:/Users/doudou/Pictures/Braggolach.jpg"
path = "C:/Users/doudou/Pictures/CLEVR.png"

def readImage(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return np.array(image)/256

def FilterEmpty(List):
    returnList = []
    for i in range(len(List)):
        if (List[i] != ""):
            returnList.append(List[i])
    return returnList

image = readImage(path)

# setup basic configuration
word_dim =  1 # the representation dimension of words
state_dim = 128 # the representation dimension of semantics
ops_dim = 128 # represenatation dimension of operators
arg_dim = 64 # representation dimension of arguments

max_len = 30 # The maximum length of a sentence
EPOCH = 2400  # training epoch shown here

class DSL_Base:
    def __init__(self):
        self.size = 0
        self.operators = ["root","whiteBoard","drawCircle","drawRectangular","Count","Union","Intersection","X","Y","Int256"]
        self.ops = [1,2,3,4,5,6,7,8,9] # Notice that root is not considered in this place
        self.arged_ops = ["Count","Union","Intersection","drawCircle"] # these are operators with arguments
        self.arguments = [0,1,2,3,4,5] # notice that arg 0 is the root argumnet
        self.arg_dict = {"Count":[1],"Union":[2,3],"Intersection":[4,5],"drawCircle":[5,6,7,8]}   # argument dictionary from operator to arguments
        
    def register_operator(self,op):
        return 0

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(4,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2,2,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out
    
def image_loss(x,img):
    d = x - img
    d2 = d * d
    for i in range(3):
        d2 = d2.sum(-1)
    return d2

class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope


def convert_to_data(x):
    x = torch.tensor(x)
    vals = torch.tensor(x.permute(2,0,1)).to(torch.float32)
    return (vals.to(torch.float32)).reshape(1,3,320,480)

def toImage(x):
    return x[0].permute(1,2,0).detach().numpy()


class MultiAttention(nn.Module):
    def __init__(self,w = 320,h = 480,n = 5):
        super().__init__()
        self.attentionMarker = AttentionNet()
        self.masks = []
        self.w = w
        self.h = h
        self.iters = n
    def forward(self,x):
        scope = torch.ones([1,1,self.w,self.h])
        for i in range(self.iters):
            mask,newscope = self.attentionMarker(x,scope)
            self.masks.append(mask)
            scope = newscope
        self.masks.append(scope)
        return self.masks
    
class EncoderNet(nn.Module):
    def __init__(self, width, height,channel):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channel, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

    
def visualizeMasks(masks):
    for i in range(len(masks)):
        plt.imshow(toImage(masks[i]),"bone")
        plt.pause(0.01)
        plt.cla()

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
                operator = ops_dict[index+1]
            
                
            self.program += str(operator)
            if (operator == "whiteBoard"):
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
        
    
c = Conceptualizer()
x = convert_to_data(image)

optimizer =torch. optim.Adam(c.parameters(), lr=1e-4)
"""
for i in range(180):
    p,Loss = c.trainPair(x,"drawCircle(whiteBoard(),1.45,2.56,0.5)")
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    print("Loss is ",Loss.detach().numpy())

print(c.representation(x))
d = Deconstructor()

"""
sf = stochField()
#print(d.destruct(x))

samples = [
    "drawCircle(whiteBoard(),1.4683618545532227,2.1186287879943848,.70646553039551)",
    "drawCircle(whiteBoard(),2.4683618545532227,1.1186287879943848,.50646553039551)",
    "drawCircle(whiteBoard(),3.4683618545532227,1.7186287879943848,.70646553039551)",
    "drawCircle(whiteBoard(),1.4683618545532227,1.1186287879943848,.50646553039551)",
    "drawCircle(whiteBoard(),3.4683618545532227,2.1186287879943848,.20646553039551)",
    "drawCircle(whiteBoard(),1.4683618545532227,1.1186287879943848,.70646553039551)",
    "drawCircle(whiteBoard(),3.4683618545532227,3.1186287879943848,.20646553039551)",
    "drawCircle(whiteBoard(),2.4683618545532227,2.7186287879943848,.40646553039551)",
    "drawCircle(whiteBoard(),4.4683618545532227,3.1186287879943848,.30646553039551)",
    "drawCircle(whiteBoard(),2.4683618545532227,2.1186287879943848,.40646553039551)",
    "drawCircle(whiteBoard(),1.2683618545532227,1.7186287879943848,.40646553039551)",
    "drawCircle(whiteBoard(),1.7683618545532227,1.1186287879943848,.20646553039551)",
    "drawCircle(whiteBoard(),2.0683618545532227,3.1186287879943848,.60646553039551)"
    ]


def realize(program):
    x = torch.tensor(eval(program))/256
    x = x.reshape([1,320,480,3])
    x = x.permute([0,3,1,2]).to(torch.float32)
    return x

for i in range(1500):
    Loss = 0
    for j in range(len(samples)):    
        x= realize(samples[j])
        
        p,loss = c.trainPair(x,samples[j])
        optimizer.zero_grad()
        Loss += loss
        
    Loss.backward()
    optimizer.step()
    print("Loss is ",Loss.detach().numpy())
 
plt.imshow(eval(act))
    