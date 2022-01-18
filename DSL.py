# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:24:17 2022

@author: doudou
"""
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