# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:33:29 2022

@author: doudou
"""
from Boomsday.datatypes import *
# using version space algebra to refractor components in programs

root = Node("LilRag")

n2 = Node("val1")
n2.attachFrom(root, "l1")
n1 = Node("val2")
n1.attachFrom(root, "l2")

n3 = Node("val3")
n3.attachFrom(n1, "l3")

n4 = Node("val4")
n4.attachFrom(n1,"l4")

n5 = Node("val5")
n5.attachFrom(n2,"l5")

t = Tree(root)

plotTree(t)