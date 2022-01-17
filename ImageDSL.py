import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random_integers
from PIL import Image

boardShape = [64,64,3]

coverMode = True

def whiteBoard():
    return np.zeros(boardShape)

def drawCircle(board,xCenter,yCenter,radius,color = 0):
    if (coverMode):
        sprite = board
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                distCenterSquare = (x - xCenter)**2 + (y-yCenter)**2
                if(distCenterSquare<radius ** 2):
                    sprite[x][y][color] = 1.0
        return sprite
    else:
        sprite = np.zeros(boardShape)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                distCenterSquare = (x - xCenter)**2 + (y-yCenter)**2
                if(distCenterSquare<radius ** 2):
                    sprite[x][y][color] = 1.0
        return sprite + board


def drawRectangle(board,xCenter,yCenter,width,height,color = 1):
    if (coverMode):
        sprite = board
        sprite[xCenter:xCenter+width,yCenter:yCenter+height,color] = 1.0
        return sprite + board
    else:
        sprite = np.zeros(boardShape)
        sprite[xCenter:xCenter+width,yCenter:yCenter+height,color] = 1.0
        return sprite + board
    
def drawRectangle(board,xCenter,yCenter,width,height,color = 1):
    if (coverMode):
        sprite = board
        sprite[xCenter:xCenter+width,yCenter:yCenter+height,color] = 1.0
        return sprite + board
    else:
        sprite = np.zeros(boardShape)
        sprite[xCenter:xCenter+width,yCenter:yCenter+height,color] = 1.0
        return sprite + board
    