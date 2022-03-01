import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import lr_scheduler
import sys
import h5py
import random
import copy
from matplotlib import pyplot as plt
from PIL import Image
import time
from torch.utils.data import Dataset,random_split
from torch import optim
from time import time
from torchvision import transforms
import glob
import math
import xlwt
import xlrd                           #导入模块
from xlutils.copy import copy
import torch
from torch import nn
from torch.nn import functional as F
import random


def read_data(datapath,labelpath,began):
    data = np.load(datapath)
    label = np.load(labelpath)
  
    del_list=[]
    for i in range(began,label.shape[0]):
        if label[i].max()!=1:
            del_list.append(i)
    data = np.delete(data, del_list, axis=0)
    label = np.delete(label, del_list, axis=0)
    return data,label

    
def lrfliplr(x):
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
        return x

class Traindata_Loader(Dataset):
    def __init__(self):
        self.data,self.label = read_data("/Share/home/10014/***/h6/data.npy"
                                         ,"/Share/home/10014/***/h6/label.npy",40*189)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=32)
        
    def __getitem__(self, index):
        #  特征和标签图片路径
        p1 = random.randint(0,1) 
        p2 = random.randint(0,1)
        data = self.data[index]
        label = self.label[index]
        
        if p1 == 100:
            dataAug=np.ascontiguousarray(np.fliplr(data))
            labelAug = np.ascontiguousarray(np.fliplr(label))
            max = np.expand_dims(labelAug, 0)
            # max=np.expand_dims(max,0)
            max = torch.from_numpy(max)
            #print(max.shape)
            max = self.MaxPool2d(max)
            reshape_size = (1, 1, 1, 36)
            max = torch.reshape(max, reshape_size)
            max = torch.squeeze(max)
            print("上下")
            #return torch.from_numpy(dataAug),torch.from_numpy(labelAug),max
            return dataAug,labelAug,max
        if p2 == 100:
            dataAug = np.ascontiguousarray(np.transpose(np.fliplr(np.transpose(data, (0, 2, 1))), (0, 2, 1)))
            labelAug = np.ascontiguousarray(np.transpose(np.fliplr(np.transpose(label, (0, 2, 1))), (0, 2, 1)))
            max = np.expand_dims(labelAug, 0)
            # max=np.expand_dims(max,0)
            max = torch.from_numpy(max)
            #print(max.shape)
            max = self.MaxPool2d(max)
            reshape_size = (1, 1, 1, 36)
            max = torch.reshape(max, reshape_size)
            max = torch.squeeze(max)
            print("左右")
            #return torch.from_numpy(dataAug),torch.from_numpy(labelAug),max
            return dataAug,labelAug,max

        max = np.expand_dims(label, 0)
        # max=np.expand_dims(max,0)
        max = torch.from_numpy(max)
        #print(max.shape)
        max = self.MaxPool2d(max)
        reshape_size = (1, 1, 1, 36)
        max = torch.reshape(max, reshape_size)
        max = torch.squeeze(max)
        # print(max.shape)
        # 读取训练图片和标签图片
        return data, label,max

    def __len__(self):
        # 返回训练集大小
        return self.data.shape[0]

class Testdata_Loader(Dataset):
    def __init__(self):
        self.data = np.load("/Share/home/10014/***/h6/test_data.npy")
        self.label = np.load("/Share/home/10014/***/h6/test_label.npy")
    def __getitem__(self, index):
        #  特征和标签图片路径
        data = self.data[index]
        label = self.label[index]

        return data, label 
    def __len__(self):
        # 返回训练集大小
        return self.data.shape[0]
