# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:18:06 2022

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
# TODO: node tree structure

from config import *
from DSL import *
from image_net import *
from weaver import *


path = "C:/Users/doudou/Pictures/CLEVR.png"

# read the image and convert it to the form of RGB/256
def readImage(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return np.array(image)/256


def convert_to_data(x):
    x = torch.tensor(x)
    vals = torch.tensor(x.permute(2,0,1)).to(torch.float32)
    return (vals.to(torch.float32)).reshape(1,3,320,480)

def toImage(x):
    return x[0].permute(1,2,0).detach().numpy()

def realize(program):
    x = torch.tensor(eval(program))/256
    x = x.reshape([1,320,480,3])
    x = x.permute([0,3,1,2]).to(torch.float32)
    return x


image = readImage(path)
c = Conceptualizer()
# torch.load("concept.pth")
x = convert_to_data(image)

optimizer =torch. optim.Adam(c.parameters(), lr=1e-4)


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
    "drawCircle(whiteBoard(),2.0683618545532227,2.1186287879943848,.40646553039551)",
    "drawCircle(whiteBoard(),1.6683618545532227,2.1186287879943848,.40646553039551)",
    
    "drawRectangle(whiteBoard(),1.6683618545532227,2.1186287879943848,.40646553039551,0.35)",
    "drawRectangle(whiteBoard(),1.6683618545532227,2.1186287879943848,.32,0.2)",
    "drawRectangle(whiteBoard(),3.6683618545532227,1.1186287879943848,.40646553039551,0.35)",
    "drawRectangle(whiteBoard(),2.6683618545532227,1.4186287879943848,.32,0.2)",
    "drawRectangle(whiteBoard(),1.8683618545532227,1.3186287879943848,.3239551,0.25)",
    "drawRectangle(whiteBoard(),1.2683618545532227,3.1186287879943848,.32,0.2)",
    "drawRectangle(whiteBoard(),3.6683618545532227,2.4186287879943848,.40646553039551,0.35)",
    "drawRectangle(whiteBoard(),2.6683618545532227,3.4186287879943848,.32,0.2)",
    "drawRectangle(whiteBoard(),1.8683618545532227,1.3186287879943848,.1239551,0.15)",
    "drawRectangle(whiteBoard(),1.2683618545532227,3.1186287879943848,.12,0.23)",
    "drawRectangle(whiteBoard(),3.6683618545532227,2.4186287879943848,.30646553039551,0.15)",
    "drawRectangle(whiteBoard(),2.6683618545532227,3.4186287879943848,.12,0.2)"
    
    ]


for i in range(1500):
    Loss = 0
    for j in range(len(samples)):    
        x= realize(samples[j])
        
        p,loss = c.trainPair(x,samples[j])
        optimizer.zero_grad()
        Loss += loss
        
    Loss.backward()
    optimizer.step()
    print(i,"Loss is ",Loss.detach().numpy())
    torch.save(c,"concept.pth")
 
plt.imshow(eval(act))
    