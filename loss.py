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


class target_aware_loss(nn.Module):
    def __init__(self, patch_size=2, alpha=0.75):
        super(target_aware_loss, self).__init__()

        self.MaxPool2d = nn.MaxPool2d(kernel_size=patch_size)
        self.up = nn.Upsample(scale_factor=patch_size, mode='nearest')
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        gamma = 2.0
        epsilon = 1e-7
        max_map = self.MaxPool2d(y_true)
        up_map = self.up(max_map)
        alpha_map = torch.where(torch.eq(up_map, 1), torch.ones_like(y_pred) * self.alpha,
                                torch.ones_like(y_pred) * (1 - self.alpha))

        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        alpha_map = alpha_map.view(-1)

        # focal_loss
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)  # 最小值时1e-7
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        loss = -torch.mean(self.alpha * torch.pow(1.0 - pt_1, gamma) * torch.log(pt_1)) - torch.mean(
            alpha_map * torch.pow(pt_0, gamma) * torch.log(1.0 - pt_0))

        return loss




class train_loss(nn.Module):
    def __init__(self, patch_size = 2, alpha = 0.75):
        super(train_loss, self).__init__()
        self.target_aware_loss = target_aware_loss(patch_size,alpha)

    def forward(self, y_pred, y_true):

        ta_loss = self.target_aware_loss(y_pred, y_true)

        smooth = 1.
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

        return ta_loss - torch.log(dice_loss)