# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:24:17 2022

@author: doudou
"""
class DSL_Base:
    def __init__(self):
        self.size = 0
        self.argSize = 1
        self.operators = ["root"]
        #["root","whiteBoard","drawCircle","drawRectangle","Count","Union","Intersection","X","Y","Int256"]
        self.ops = []
        #[1,2,3,4,5,6,7,8,9] # Notice that root is not considered in this place
        self.returnTypes = []
        #["Void"]
        self.arged_ops = []
        #["Count","Union","Intersection","drawCircle","drawRectangle"] # these are operators with arguments
        self.arguments = [0,1,2,3,4,5] # notice that arg 0 is the root argumnet
        self.arg_dict = {}
        #{"Count":[1],"Union":[2,3],"Intersection":[4,5],"drawCircle":[5,6,7,8],"drawRectangle":[9,10,11,12]}   # argument dictionary from operator to arguments
        
    def register_operator(self,op):
        try:
            self.operators.append(op.functionName)
            self.returnTypes.append(op.returnType)
            self.size += 1
            self.ops.append(self.size)
            
            if (len(op.args)>0):
                self.arg_dict[op.functionName] = []
                self.arged_ops.append(op.functionName)
                for i in range(len(op.args)):
                    self.arg_dict[op.functionName].append(self.argSize)
                    self.argSize += 1
        except:
            print("Failed to register the operator in the current DSL")
            return -1
        
    def registerOperators(self,operators):
        for op in operators:
            self.register_operator(op)
                
    
    # what we need: 
    # function nam: 
    # return type
    # {arg name, arg type}

class operator:
    def __init__(self,funcName,returnType,args):
        self.functionName = funcName
        self.returnType = returnType
        self.args = args
        
 #["root","whiteBoard","drawCircle","drawRectangle","Count","Union","Intersection","X","Y","Int256"]
prim1 = operator("Count","Int",[["Set","Set"]])
prim2 = operator("Union","Set",[["Set1","Set"],["Set2","Set"]])    
prim3 = operator("Intersection","Set",[["Set1","Set"],["Set2","Set"]])        
prim4 = operator("X","Set",[]) 
prim5 = operator("Y","Set",[])
prim6 = operator("whiteBoard","Image",[])
prim7 = operator("drawCircle","Image",[["image","Image"],["x","Int"],["y","Int"],["r","Int"]])
prim8 = operator("drawRectangle","Image",[["image","Image"],["x","Int"],["y","Int"],["w","Int"],["h","Int"]])
prim9 = operator("Int256","Int",[])
ops = [prim1,prim2,prim3,prim4,prim5,prim6,prim7,prim8,prim9]
DSL = DSL_Base()

DSL.registerOperators(ops)       