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


from net import *
from data import *
from loss import *


def metrics(pred, gt):
    pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum(gt_flat * pred_flat)
    fp = torch.sum(pred_flat) - tp
    fn = torch.sum(gt_flat) - tp
    return tp, fp, fn


def evolution(data_loader, net, model_path):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    people_tp = 0
    people_fp = 0
    people_fn = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    num = 0
    dice_list = []
    recall_list = []
    precision_list = []

    with torch.no_grad():
        for index, (x, y) in enumerate(data_loader):
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            y = torch.as_tensor(y, dtype=torch.float32).to(device)
            output_val, out4, out3, out2, out1, we1, we2, we3, we4, we5 = net(x)
            tp, fp, fn = metrics(output_val.cpu(), y.cpu())
            people_fn += fn
            people_fp += fp
            people_tp += tp
            num += 21
            if num == 189:
                recall_per = people_tp / (people_tp + people_fn)
                precision_per = people_tp / (people_tp + people_fp)
                if np.isnan(precision_per.cpu()):
                    precision_per = 0
                dice_per = (2 * people_tp / (2 * people_tp + people_fp + people_fn))
                dice_list.append(dice_per)
                recall_list.append(recall_per)
                precision_list.append(precision_per)
                total_fn += people_fn
                total_tp += people_tp
                total_fp += people_fp
                people_fn = 0
                people_fp = 0
                people_tp = 0
                num = 0
        dice_all = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        dice_list = sorted(dice_list)
        dice_mean = np.mean(dice_list)
        dice_std = np.std(dice_list)
    return dice_all, dice_list, recall_list, precision_list


if __name__ == "__main__":

    model_path = '/Share/home/10014/****/third-2/net.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add
    print(device)

    val_dataset = Testdata_Loader()
    print("the total num of test data:", len(val_dataset))
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=21, num_workers=0)

    net = EUnet(in_channels=4).to(device)
    dice_all, dice_list, recall_list, precision_list = evolution(val_loader, net, model_path)
    print(str(dice_all.numpy()),
          str(np.mean(dice_list))+ '±' + str(np.std(dice_list, ddof=1)),
    str(np.mean(recall_list)) + '±' + str(np.std(recall_list, ddof=1)),
    str(np.mean(precision_list)) + '±' + str(np.std(precision_list, ddof=1)))
