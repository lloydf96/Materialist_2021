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
from image_dataset import *
from segmentation_unet import *
from train import *

if __name__ == "__main__":
    
    '''
    Here we train the model three times
    1. We train the upsample phase of the UNET
    2. We train the upsample and the last two resnet blocks of the downsample UNET
    3. We train the first three resnet blocks of the downsample unet
    '''
    
    batch_size = 16
    num_workers = 4
    sample_per_class = 4000
    n_epochs = 10
    accumulate_steps = 4
    
    train_file_location =os.path.join(os.getcwd(),'train','train.csv')
    train_image_location =os.path.join(os.getcwd(),'train')

    dev_file_location =os.path.join(os.getcwd(),'dev','dev.csv')
    label_location =dev_file_location
    dev_image_location =os.path.join(os.getcwd(),'dev')

    train_set = pd.read_csv(os.path.join(train_file_location))
    dev_set = pd.read_csv(os.path.join(dev_file_location))
    train_set['ClassId'] = train_set['ClassId'] + 1
    dev_set['ClassId'] = dev_set['ClassId'] + 1

    train_class_tbl = train_set.ClassId.drop_duplicates()
    dev_class_tbl = dev_set.ClassId.drop_duplicates()

    train_ds = ImageDataset((512,512),train_file_location,label_location,train_image_location,sample_per_class,train_class_tbl)
    dev_ds = ImageDataset((512,512),dev_file_location,label_location,dev_image_location,100,dev_class_tbl)
    
    dev_dl = torch.utils.data.DataLoader(dataset=dev_ds, shuffle=True, batch_size=batch_size,num_workers = num_workers)
    op_layers = len(train_class_tbl)
    net = UNet(op_layers)
    
    focal_loss = FocalLoss()
  
    device = torch.device("cuda")
    net.to(device)
    #-------------------------------------------------------Train once ------------------------------------------------------------------------------------------
    
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
            
    for name, layer in net.named_modules():
        if name.startswith('dn'):
            layer.requires_grad= False
        else:
            layer.requires_grad = True
            
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs,eta_min = 1e-6)
    train_losses,test_losses = fit_and_evaluate(net, optimizer, focal_loss,scheduler, train_ds ,dev_dl, n_epochs, 2,'model_1_', batch_size=batch_size,num_workers = num_workers)
    
    # ------------------------------------------------------ Retrain ---------------------------------------------------------------------------------------------
    checkpoint = torch.load(os.path.join(os.getcwd(),'models','model_1_7.h5py'))
    net.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    n_epochs = 10
    
    for name, layer in net.named_modules():
        if name.startswith(('dnlayer64x128','dnconvlayer64x256','dnlayer128x64')):
            layer.requires_grad= False
        else:
            layer.requires_grad = True
        
    optimizer = optimizer = optim.Adam(net.parameters(), lr=5e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs,eta_min = 5e-7)
                           
    train_losses,test_losses = fit_and_evaluate(net, optimizer, focal_loss,scheduler, train_ds ,dev_dl, n_epochs,2,'model_2_', batch_size=batch_size,num_workers = num_workers)
    
    # ------------------------------------------------------ Retrain --------------------------------------------------------------------------------------------------
    checkpoint = torch.load(os.path.join(os.getcwd(),'models','model_2_7.h5py'))
    net.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    n_epochs = 10
    
    for name, layer in net.named_modules():
        if name.startswith(('dnlayer64x128','dnconvlayer64x256','dnlayer128x64')):
            layer.requires_grad= True
        else:
            layer.requires_grad = False
        
    optimizer = optimizer = optim.Adam(net.parameters(), lr=5e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs,eta_min = 5e-8)
                           
    train_losses,test_losses = fit_and_evaluate(net, optimizer, focal_loss,scheduler, train_ds ,dev_dl, n_epochs,2,'model_3_', batch_size=batch_size,num_workers = num_workers)
        
                      
    