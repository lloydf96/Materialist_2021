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
from segmentation_unet import *

def log_statement(statement):
    text_file = open("log.txt", "a")

    n = text_file.write('\n'+statement)
    text_file.close()
        
def fit_and_evaluate(net, optimizer, loss_func, scheduler, train_ds ,dev_dl, n_epochs,accumulate_steps,model_save_name, batch_size=1,num_workers = 1 ):
    
    train_losses = []
    dev_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loss_func = loss_func.to(device)
    num_classes = train_ds.class_len+1
    train_len = len(train_ds) 
    dev_len = len(dev_ds)

    print("epochs begin")
    for i in range(n_epochs):
        id_list = getSubsetId(train_ds)
        train_subset = torch.utils.data.Subset(train_ds,id_list.tolist())
        train_dl = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size,pin_memory = True,num_workers = num_workers) 

        print("epoch number: ",i)
        log_statement("epoch number : "+str(i))
        
        with torch.no_grad():
            start_time = time.time()
            if i%1 == 0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func
                    }, os.path.join(os.getcwd(),'models',model_save_name+str(i)+'.h5py'))
                
            cum_loss = 0
        
        net.train()
        
        optimizer.zero_grad() #comment this line for op
        
        for step,(train_x,train_y) in enumerate(train_dl):
            
            train_x,train_y = train_x.to(device),train_y.to(device)
            net_output= net(train_x)
            loss = loss_func(net_output, train_y)
            #divide by accumulate_steps so that the loss is normalized and gradient is normalized accordingly
            loss = loss/accumulate_steps
            
            loss.backward()
            
            if ((step+1)%accumulate_steps == 0):
                optimizer.step()
                net.zero_grad()
                
                with torch.no_grad():
                    cum_loss += loss.detach()
                        
        with torch.no_grad():
            print("for %d step, time_elapsed for epoch %d is %d" %(step,i, time.time() - start_time))
            
        scheduler.step()
        
        if (i%1== 0):
            net.eval()
            
            with torch.no_grad():
                
                #train_risk,train_iou = epoch_loss(net,loss_func,train_dl,device,num_classes,no_of_step = 10)
                dev_risk,dev_iou= epoch_loss(net, loss_func, dev_dl,device,num_classes)
                train_losses.append((i+1,cum_loss))
                dev_losses.append((i+1,dev_risk.detach()))
                
                print("Epoch: %s, Training loss: %s, Testing loss: %s " %(i,cum_loss/(step),dev_risk))
                print("Epoch: %s,  Testing iou: %s " %(i,dev_iou))
                
                log_statement("Epoch: %s, Training loss: %s, Testing loss: %s " %(i,cum_loss/(step),dev_risk))
                log_statement("Epoch: %s,  Testing iou: %s " %(i,dev_iou))

            net.train()
        
    return train_losses, dev_losses

def iou_score(predictions, target,num_classes):
    ep = 10e-7
    predictions = mask_to_one_hot(predictions, num_classes )
    target = F.one_hot(target,num_classes = num_classes ).permute([0,3,1,2])
    denominator = predictions.type(torch.float) + target.type(torch.float)
    #denominator = denominator[:,target.unique(),:,:]
    numerator = (denominator >1).type(torch.float)
    denominator = (denominator >0.5).type(torch.float)

    iou_list = (numerator.sum(axis = 2).sum(axis = 2)+ep)/(denominator.sum(axis = 2).sum(axis = 2)+ep)
    iou = iou_list.mean(dim = 1).sum()
    
    return iou

def mask_to_one_hot(img,num_classes):
    img = torch.argmax(img,axis = 1)
    img = F.one_hot(img,num_classes = num_classes ).permute([0,3,1,2])
    return img.type(torch.float)

def epoch_loss(net, loss_func, dl,device,num_classes):

    loss = 0   
    iou = 0
    len_dl = 0
    for x,y in dl:
        x,y = x.to(device),y.to(device)

        y_op= net(x).detach()
        loss += loss_func(y_op,y).detach()
        iou += iou_score(y_op, y,num_classes).detach()
        len_dl += x.shape[0]


    return  loss/len_dl,iou/len_dl

def getSubsetId(dataset):
    metadata =dataset.metadata.copy()
    metadata.index.set_names(['index'],inplace = True)
    metadata.reset_index(inplace = True)
    g =metadata.groupby('ClassId')
    sample_per_class = min(g.size().min(),dataset.sample_per_class)
    subset_id = g.apply(lambda group: group.sample(sample_per_class)).reset_index(drop = True)['idx'].drop_duplicates()
    return subset_id

