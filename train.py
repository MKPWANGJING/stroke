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


def diceCoeffv2(pred, gt, eps=1e-7):
    pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() /N

def metrics(pred, gt):
    pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum(gt_flat * pred_flat)
    fp = torch.sum(pred_flat) - tp
    fn = torch.sum(gt_flat) - tp
    return tp, fp, fn


def evolution(data_loader,net,model_path):
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
            output_val,out4,out3,out2,out1,we1,we2,we3,we4,we5 = net(x)
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
    return dice_all,dice_list,recall_list,precision_list



if __name__ == "__main__":

    model_save_path = '/Share/home/10014/Makunpeng/third-2/net.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add
    print(device)
    
    train_dataset = Traindata_Loader()
    print("the total num of train data:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,num_workers=4,shuffle=True)

    net = EUnet(in_channels=4).to(device)
    criterion = train_loss(patch_size = 1, alpha = 0.25 * 2).to(device)
    criterion_attention=torch.nn.BCELoss().to(device)

    optimizer = optim.Adam(net.parameters(),weight_decay=0.00001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10,gamma = 0.96)
    
    create_excel(excel_save_path)
    for epoch in range(300):
        net.train()
        for step, (x, y , z) in enumerate(train_loader):

            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            y = torch.as_tensor(y, dtype=torch.float32).to(device)
            z = torch.as_tensor(z, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output,out4,out3,out2,out1,we1,we2,we3,we4,we5 = net(x)

            # 损失函数计算
            loss = criterion(output, y)
            # 多层监督损失函数
            loss0 = criterion(out4, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss1 = criterion(out3, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss2 = criterion(out2, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss3 = criterion(out1, y)
            # 注意力损失函数
            loss_1 = criterion_attention(we1, z)
            loss_2 = criterion_attention(we2, z)
            loss_3 = criterion_attention(we3, z)
            loss_4 = criterion_attention(we4, z)
            loss_5 = criterion_attention(we5, z)
            loss_total = loss + loss0 + loss1 + loss2 + loss3 + 0.15*(loss_1 + loss_2 + loss_3 + loss_4) + 0.5 * loss_5
            iter_loss = loss.item()
            loss_total.backward()
            optimizer.step()

        scheduler.step()

        torch.save(net.state_dict(), model_save_path)
