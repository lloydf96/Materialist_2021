import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchvision.models as models

from PIL import Image
import json
import os
import random
from scipy.sparse import csr_matrix
import time
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        logpt = - F.cross_entropy(output, target,reduction = 'none')
        pt    = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        balanced_focal_loss = (self.balance_param * focal_loss).sum(axis = 2).sum(axis = 1).mean()

        return balanced_focal_loss

    
class UNet(nn.Module):
    def __init__(self,op_layers):
      
        super().__init__()
        resnet = models.resnet34(pretrained = True)
        self.dnconvlayer64x256 =resnet.conv1
        self.dnbn1_relu = nn.Sequential(resnet.bn1,
                                        resnet.relu)
        self.dnmaxpool64x128= resnet.maxpool
        self.dnlayer64x128 = resnet.layer1
        self.dnlayer128x64 = resnet.layer2
        self.dnlayer256x32 = resnet.layer3
        self.dnlayer512x16 = resnet.layer4
        
        self.uplayer256x32_tr = self.bn_upsample(512,256) 
        self.uplayer256x32_conv = self.conv_seq(512,256)
        self.uplayer128x64_tr = self.bn_upsample(256,128)  
        self.uplayer128x64_conv = self.conv_seq(256,128)
        self.uplayer64x128_tr =self.bn_upsample(128,64)  
        self.uplayer64x128_conv = self.conv_seq(128,64)
        self.uplayer64x256_tr = self.bn_upsample(64,64)
        self.uplayer64x256_conv = self.conv_seq(128,64)
        self.uplayer64x512_tr = self.bn_upsample(64,64)
        
        self.op_layers = op_layers+1
        self.opconv = self.op_seq([64,32,32,self.op_layers])
        
    def bn_upsample(self,c1,c2,kernel = (2,2),stride = 2):
        bn_upsample = nn.Sequential(nn.BatchNorm2d(c1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ConvTranspose2d(c1,c2,kernel,stride = stride))
        return bn_upsample
                                    
    def conv_seq(self,ip_channel,channel,kernel = (3,3),padding = 1,stride = 1):
        
        conv = nn.Sequential(nn.Conv2d(ip_channel,channel,kernel,padding= padding,stride = stride),
                                      nn.ReLU(),

                                      nn.Conv2d(channel,channel,kernel,padding = padding,stride = stride),
                                      nn.ReLU(),

                                      nn.Conv2d(channel,channel,kernel,padding = padding,stride = stride),
                                      nn.ReLU())
        return conv
    
    def op_seq(self,channel_list,kernel = (3,3),padding = 1,stride = 1):
        
        ip_channel = channel_list[0]
        channel_1 = channel_list[1]
        channel_2 = channel_list[2]
        channel_3 = channel_list[3]
        conv = nn.Sequential(nn.Conv2d(ip_channel,channel_1,kernel,padding= padding,stride = stride),
                                      nn.ReLU(),

                                      nn.Conv2d(channel_1,channel_2,kernel,padding = padding,stride = stride),
                                      nn.ReLU(),

                                      nn.Conv2d(channel_2,channel_3,kernel,padding = padding,stride = stride),
                                      nn.ReLU())
        return conv
    
    def normalize(self,img):
        
        batch_size = img.shape[0]
        img_view = img.view(batch_size,3,-1)
        img_std,_ = img_view.max(dim = 2)
        img = img/img_std[:,:,None,None]
        
        return img
    
    def forward(self, x):
        x = self.normalize(x)
        convlayer64x256 = self.dnconvlayer64x256(x)
        bn1_relu = self.dnbn1_relu(convlayer64x256)
        maxpool64x128 = self.dnmaxpool64x128(bn1_relu)
      
        dnlayer64x128 = self.dnlayer64x128(maxpool64x128)
        dnlayer128x64 = self.dnlayer128x64(dnlayer64x128)
        dnlayer256x32 = self.dnlayer256x32(dnlayer128x64)
        dnlayer512x16 = self.dnlayer512x16(dnlayer256x32)

        uplayer256x32_tr = self.uplayer256x32_tr(dnlayer512x16)
        uplayer512x32_cat = torch.cat([uplayer256x32_tr,dnlayer256x32],dim = 1)
        uplayer256x32_conv = self.uplayer256x32_conv(uplayer512x32_cat)

        uplayer128x64_tr = self.uplayer128x64_tr(uplayer256x32_conv)
        uplayer256x64_cat = torch.cat([uplayer128x64_tr,dnlayer128x64],dim = 1)
        uplayer128x64_conv = self.uplayer128x64_conv(uplayer256x64_cat)

        uplayer64x128_tr = self.uplayer64x128_tr(uplayer128x64_conv)
        uplayer128x128_cat = torch.cat([uplayer64x128_tr,dnlayer64x128],dim = 1)
        uplayer64x128_conv = self.uplayer64x128_conv(uplayer128x128_cat)

        uplayer64x256_tr = self.uplayer64x256_tr(uplayer64x128_conv)
        uplayer128x256_cat = torch.cat([uplayer64x256_tr,convlayer64x256],dim = 1)
        uplayer64x256_conv = self.uplayer64x256_conv(uplayer128x256_cat)
        uplayer64x512_tr = self.uplayer64x512_tr(uplayer64x256_conv)
        
        opconv = self.opconv(uplayer64x512_tr)

        return opconv
    
    @staticmethod
    def init_he(layer):
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(layer.weight)