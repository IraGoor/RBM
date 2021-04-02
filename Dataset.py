# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:48:08 2020

@author: irago
"""

import pandas as pd
import numpy as np
import os
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image 
import re
from torch.utils import data
import math


img_folder=r'E:\study\semester_7\Data_mining\final_project\COVID-CT\CT_COVID\png'

#def findA
"""delete the generated set of pictures after the end of the program folder"""
def deleteSet(path):
        
    dirs=os.listdir(path)
    for item in dirs:
        os.remove(path + "\\" + item)
    os.rmdir(path)
"""create a folder according to a given name"""
def createFolder(name):
    if not os.path.exists('./' + name):
        os.mkdir(name)
    return os.getcwd() + '\\' + name

"""extracts label from a given a file name"""
def faceslabelFunc(filename):
    pattern =re.match(r'subject\d\d', filename)
    a=pattern.group()
    return int(a[-2:])



"""a generator for the training and test sets"""
class SetGenerator():    
    def __init__(self,image_folder,pic_size,testRatio,labelFunc=faceslabelFunc):
        self.image_folder=image_folder
        self.pic_size=pic_size
        self.testRatio=testRatio
        self.divade_datasets(labelFunc)
        
    
        
    def get_training_set(self):
        return self.training_set
    
    def get_test_set(self):
        return self.test_set
    
    def divade_datasets(self,labelFunc):
        dirs = os.listdir( self.image_folder )
        test_size=math.ceil(len(dirs) * self.testRatio)
        test_List=[]
        for i in range(test_size):
             item=random.choice(dirs)
             dirs.remove(item)
             test_List.append(item)
# =============================================================================
#         setup training set in a new folder and dataset
#             
# =============================================================================
        training_path= createFolder('training_set')
        self.setPicFormat(training_path, dirs)
        self.training_set=Dataset(training_path,labelFunc)
        
        
        test_path=createFolder('test_set')
        self.setPicFormat(test_path, test_List)
        self.test_set=Dataset(test_path,labelFunc)
        
        
        
        
        
        
        
    def setPicFormat(self,dst,dataList):
        for i,item in enumerate(dataList):
            if os.path.isfile(self.image_folder+ "\\" + item):
                im = Image.open(self.image_folder+ "\\" + item)
                im=im.convert("L")
               # f, e = os.path.splitext(self.image_folder + "\\" + item)
                imResize = im.resize(self.pic_size, Image.ANTIALIAS)
                imResize.save(dst + "\\" + item)

    def deleteSets(self):
        deleteSet(createFolder('training_set'))
        deleteSet(createFolder('test_set'))
        
        
# =============================================================================
#     def resize(img_folder,dst,dataList):
#         for item in dataList:
#             if os.path.isfile(img_folder+ "\\" + item):
#                 im = Image.open(img_folder+ "\\" + item)
#                 im=im.convert("L")
#                 f, e = os.path.splitext(img_folder + "\\" + item)
#                 imResize = im.resize((200,200), Image.ANTIALIAS)
#                 imResize.save(dst + "\\" + f +' resized.jpg')    
# 
# 
# =============================================================================
        
"""implemantion of a dataset according to our requirements"""
class Dataset(data.Dataset):
    def __init__(self,image_folder,labelFunc):
        self.image_folder=image_folder
        self.vectorPixelsList,self.lables = self.convertImgToArray(labelFunc)
        
    
    def __len__(self):
        return len(self.vectorPixelsList)
    
    def __getitem__(self, index):
        return (self.vectorPixelsList[index],self.lables[index])
    
    


"""this function converts a picture file into a numpy array"""
def convertImgToArray(self,labelFunc):
        pictureList=[]
        lablesList =[]
        dirs = os.listdir( self.image_folder )
        for item in dirs:
            if os.path.isfile(self.image_folder+ "\\" + item):
                im = Image.open(self.image_folder+ "\\" + item)
                lablesList.append(labelFunc(item))
                pic =np.array(im, dtype= np.double)
                pic=pic/255
                pic=np.reshape(pic,(pic.shape[0] * pic.shape[1]))
                pictureList.append(pic)
        return pictureList,lablesList
        
            
            
            

