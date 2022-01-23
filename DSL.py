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

MathDSL = DSL_Base()

add = operator("add","Num",
               [["n1","Num"],["n2","Num"]])

subtract = operator("subtract","Num",
                    [["n1","Num"],["n2","Num"]])

multiply = operator("multiply","Num",
                    [["n1","Num"],["n2","Num"]])

divide = operator("divide","Num",
                  [["n1","Num"],["n2","Num"]])

log = operator("log","Num",
               [["n1","Num"]])

sqrt = operator("sqrt","Num",
                [["n1","Num"]])

factorial = operator("factorial","Int",
                     [["n1","Int"]])

gcd = operator("gcd","Int",
               [["n1","Int"],["n2","Int"]])

lcm = operator("lcm","Int",
               [["n1","Int"],["n2","Int"]])

power = operator("power","Num",
                 [["n1","Num"],["n2","Num"]])

maximum = operator("max","Num",
                   [["n1","Num"],["n2","Num"]])

minimum = operator("min","Num",
                   [["n1","Num"],["n2","Num"]])

reminder = operator("reminder","Num",
                    [["n1","Num"],["n2","Num"]])

negate = operator("negate","Num",
                  [["n1","Num"],["n2","Num"]])

inverse = operator("inverse","Num",
                   [["n1","Num"],["n2","Num"]])

roundNum = operator("round","Num",
                    [["n1","Num"],["n2","Num"]])

floor = operator("floor","Num",
                 [["n1","Num"],["n2","Num"]])

sine = operator("sine","Num",
                [["n1","Num"],["n2","Num"]])

cosine = operator("cosine","Num",
                  [["n1","Num"],["n2","Num"]])

tangent = operator("tangent","Num",
                   [["n1","Num"],["n2","Num"]])

radians_to_degree = operator("radians_to_degree","Num",
                             [["n1","Num"],["n2","Num"]]) 

degree_to_radians = operator("degree_to_radians","Num",
                             [["n1","Num"],["n2","Num"]])

sum_consecutive_number = operator("sum_consecutive_number","Num",
                                  [["n1","Num"],["n2","Num"]])

circle_area = operator("circle_area","Num",
                       [["n1","Num"],["n2","Num"]])

circumface = operator("circumface","Num",
                      [["n1","Num"],["n2","Num"]])

circle_arc = operator("circle_arc","Num",
                      [["n1","Num"],["n2","Num"]])

semi_circle_perimiter = operator("semi_circle_perimiter","Num",
                                 [["n1","Num"],["n2","Num"]])

circle_sector_area = operator("circle_sector_area","Num",
                              [["n1","Num"],["n2","Num"]])

rectangle_perimeter = operator("rectangle_perimeter","Num",
                               [["n1","Num"],["n2","Num"]])

rectangle_area = operator("rectangle_area","Num",
                          [["n1","Num"],["n2","Num"]])

square_perimeter = operator("square_perimeter","Num",
                            [["n1","Num"],["n2","Num"]])

square_area = operator("square_area","Num",
                       [["n1","Num"],["n2","Num"]])

trapezium_area = operator("trapezium_area","Num",
                          [["n1","Num"],["n2","Num"]])

rhombus_perimeter = operator("rhombus_perimeter","Num",
                             [["n1","Num"],["n2","Num"]])

rhombus_area = operator("rhombus_area","Num",
                        [["n1","Num"],["n2","Num"]])

quadrilateral_area = operator("quadrilateral_area","Num",
                              [["n1","Num"],["n2","Num"]])

volume_cone = operator("volume_cone","Num",
                       [["n1","Num"],["n2","Num"]])

volume_rectangular_prism = operator("volume_rectangular_prism","Num",
                                    [["n1","Num"],["n2","Num"]])

volume_cube = operator("volume_cube","Num",
                       [["n1","Num"],["n2","Num"]])

volume_sphere = operator("volume_sphere","Num",
                         [["n1","Num"],["n2","Num"]])

volume_cylinder = operator("volume_cylinder","Num",
                           [["n1","Num"],["n2","Num"]])

surface_cone= operator("surface_cone","Num",
                       [["n1","Num"],["n2","Num"]])

surface_cylinder= operator("surface_cylinder","Num",
                           [["n1","Num"],["n2","Num"]])

surface_cube= operator("surface_cube","Num",
                       [["n1","Num"],["n2","Num"]])

surface_rectangular_prism= operator("surface_rectangular_prism","Num",
                                    [["n1","Num"],["n2","Num"]])

surface_sphere = operator("surface_sphere","Num",
                          [["n1","Num"],["n2","Num"]])

side_by_diagonal = operator("side_by_diagonal","Num",
                            [["n1","Num"],["n2","Num"]])

cube_edge_by_volume = operator("cube_edge_by_volume","Num",
                               [["n1","Num"],["n2","Num"]])

diagonal = operator("diagonal","Num",
                    [["n1","Num"],["n2","Num"]])

square_edge_by_perimeter = operator("square_edge_by_perimeter","Num",
                                    [["n1","Num"],["n2","Num"]])

square_edge_by_area = operator("square_edge_by_area","Num",
                               [["n1","Num"],["n2","Num"]])

triangle_perimeter = operator("triangle_perimeter","Num",
                              [["n1","Num"],["n2","Num"]])

triangle_area = operator("triangle_area","Num",
                         [["n1","Num"],["n2","Num"]])

triangle_area_three_edges = operator("triangle_area_three_edges","Num",
                                     [["n1","Num"],["n2","Num"]])

union_prob = operator("union_prob ","Num",
                      [["n1","Num"],["n2","Num"]])

negate_prob = operator("negate_prob","Num",
                       [["n1","Num"],["n2","Num"]])

combination = operator("combination","Num",
                       [["n1","Num"],["n2","Num"]])

permutation = operator("permutation","Num",
                       [["n1","Num"],["n2","Num"]])

count_interval = operator("count_interval","Num",
                          [["n1","Num"],["n2","Num"]])

percent = operator("percent","Num",
                   [["n1","Num"],["n2","Num"]])

p_after_gain = operator("p_after_gain","Num",
                        [["n1","Num"],["n2","Num"]])

p_after_loss = operator("p_after_loss","Num",
                        [["n1","Num"],["n2","Num"]])

price_after_gain = operator("price_after_gain","Num",
                            [["n1","Num"],["n2","Num"]])

price_after_loss = operator("price_after_loss","Num",
                            [["n1","Num"],["n2","Num"]])

from_percent = operator("from_percent","Num",
                        [["n1","Num"],["n2","Num"]])

gain_percent = operator("gain_percent","Num",
                        [["n1","Num"],["n2","Num"]])

loss_percent = operator("loss_percent","Num",
                        [["n1","Num"],["n2","Num"]])

negate_percent = operator("negate_percent","Num",
                          [["n1","Num"],["n2","Num"]])

original_price_before_gain = operator("original_price_before_gain","Num",
                                      [["n1","Num"],["n2","Num"]])

original_price_before_loss = operator("original_price_before_loss","Num",
                                      [["n1","Num"],["n2","Num"]])

to_percent = operator("to_percent","Num",
                      [["n1","Num"],["n2","Num"]])

speed = operator("speed","Num",
                 [["n1","Num"],["n2","Num"]])

combined_work = operator("combined_work","Num",
                         [["n1","Num"],["n2","Num"]])

find_work = operator("find_work","Num",
                     [["n1","Num"],["n2","Num"]])

speed_ratio_steel_to_stream = operator("speed_ratio_steel_to_stream","Num",
                                       [["n1","Num"],["n2","Num"]])

speed_in_still_water = operator("speed_in_still_water","Num",
                                [["n1","Num"],["n2","Num"]])

stream_speed = operator("stream_speed","Num",
                        [["n1","Num"],["n2","Num"]])
# define some constants to use 
CONST_pi = operator("const_pi","Num",[])
CONST_2 = operator("const_2","Num",[])
CONST_1 = operator("const_1","Num",[])
CONST_3 = operator("const_3","Num",[])
CONST_4 = operator("const_4","Num",[])
CONST_6 = operator("const_6","Num",[])
CONST_10 = operator("const_10","Num",[])
CONST_100 = operator("const_100","Num",[])
CONST_1000 = operator("const_1000","Num",[])
CONST_60 = operator("const_60","Num",[])
CONST_3600 = operator("const_3600","Num",[])
CONST_16 = operator("const_1.6","Num",[])
CONST_06 = operator("const_0.6","Num",[])
CONST_02778 = operator("const_0.2778","Num",[])
CONST_03937 = operator("const_0.3937","Num",[])
CONST_254 = operator("const_2.54","Num",[])
CONST_04535 = operator("const_0.4535","Num",[])
CONST_22046 = operator("const_2.2046","Num",[])
CONST_36 = operator("const_3.6","Num",[])
CONST_DEG_TO_RAD = operator("const_DEG_TO_RAD","Num",[])
CONST_180 = operator("const_180","Num",[])
CONST_025 = operator("const_0.25","Num",[])
CONST_033 = operator("const_0.33","Num",[])

MathOps = [add,
subtract,
multiply,
divide,
log,
sqrt,
factorial,
gcd,
lcm,
power,
maximum,
minimum,
reminder,
negate,
inverse,
roundNum,
floor,
sine,
cosine,
tangent,
radians_to_degree,
degree_to_radians,
sum_consecutive_number,
circle_area,
circumface,
circle_arc,
semi_circle_perimiter,
circle_sector_area,
rectangle_perimeter,
rectangle_area,
square_perimeter,
square_area,
trapezium_area,
rhombus_perimeter,
rhombus_area,
quadrilateral_area,
volume_cone,
volume_rectangular_prism,
volume_cube,
volume_sphere,
volume_cylinder,
surface_cone,
surface_cylinder,
surface_cube,
surface_rectangular_prism,
surface_sphere,
side_by_diagonal,
cube_edge_by_volume,
diagonal,
square_edge_by_perimeter,
square_edge_by_area,
triangle_perimeter,
triangle_area,
triangle_area_three_edges,
union_prob,
negate_prob,
combination,
permutation,
count_interval,
percent,
p_after_gain,
p_after_loss,
price_after_gain,
price_after_loss,
from_percent,
gain_percent,
loss_percent,
negate_percent,
original_price_before_gain,
original_price_before_loss,
to_percent,
speed,
combined_work,
find_work,
speed_ratio_steel_to_stream,
speed_in_still_water,
stream_speed,
CONST_pi,
CONST_2,
CONST_1,
CONST_3,
CONST_4,
CONST_6,
CONST_10,
CONST_100,
CONST_1000,
CONST_60,
CONST_3600,
CONST_16,
CONST_06,
CONST_02778,
CONST_03937,
CONST_254,
CONST_04535,
CONST_22046,
CONST_36,
CONST_DEG_TO_RAD,
CONST_180,
CONST_025,
CONST_033]

MathDSL.registerOperators(MathOps)