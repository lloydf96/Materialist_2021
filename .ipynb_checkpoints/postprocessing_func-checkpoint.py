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
from train import *
from image_dataset import *
from segmentation_unet import *

def net_op(net,x,threshold):
    ''' Generates softmax output and the area of each segmented class for a given input image'''
    
    op = net(x)
    op = F.softmax(op)
    op = (op >= threshold).type(torch.float)
    op_area= op.mean(dim = 2).mean(dim = 2)
    return op,op_area

def y_one_hot(y,num_classes):
    
    ''' Generates one hot vector for an input tensor'''
    
    y_class =  F.one_hot(y,num_classes = num_classes).permute([0,3,1,2])
    y_class =torch.max(y_class,2).values
    y_class = torch.max(y_class,2).values
    return y_class

def normalize(img):
    ''' normalize image from [0,255] to  [0,1]'''
    
    batch_size = img.shape[0]
    img_view = img.view(batch_size,3,-1)
    img_std,_ = img_view.max(dim = 2)
    img = (img)/img_std[:,:,None,None]

    return img

def clean_op(y_predict,y_op):
    '''y_predict contains the classes which random forest classifies as not being noise
    The function removes classes which do not exist in y_predict from y_op '''
    
    y_predict = np.concatenate([np.ones((y_predict.shape[0],1)),y_predict],axis = 1)
    y_predict = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(y_predict),dim = 2),dim = 2)
    y_predict_op = y_op.to('cpu')*y_predict
    y_predict_op[:,0,:,:] = (y_predict_op[:,1:,:,:].sum(axis = 1) == 0).type(torch.float)
    y_predict_op = torch.argmax(y_predict_op,axis = 1)
    return y_predict_op

def dataloader(image_location,file_name,batch_size,sample_per_class=None):
    ''' defines dataset and dataloader '''
    
    file_location = os.path.join(image_location,file_name)
    data = pd.read_csv(file_location)
    data['ClassId'] = data['ClassId'] + 1
    class_tbl = data.ClassId.drop_duplicates()
    ds = ImageDataset((512,512),file_location,image_location,image_location,sample_per_class,class_tbl)
    
    if sample_per_class is not None:
        id_list = getSubsetId(ds)
        ds = torch.utils.data.Subset(ds,id_list.tolist())
        
    dl = torch.utils.data.DataLoader(dataset=ds, shuffle=True, batch_size=batch_size)

    return ds,dl


def get_postop(y_predict,y_op):
    y_predict = np.concatenate([np.ones((y_predict.shape[0],1)),y_predict],axis = 1)
    y_predict = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(y_predict),dim = 2),dim = 2)
  
    y_predict_op = y_op.to('cpu')*y_predict
    y_predict_op[:,0,:,:] = (y_predict_op[:,1:,:,:].sum(axis = 1) == 0).type(torch.float)
    return y_predict_op

def iou_loss_postop_dl(net, dl,device,num_classes):

    loss = 0   
    iou = 0
    len_dl = 0
    
    for x,y in dl:
        
        x,y = x.to(device),y.to(device)
        y_op,y_op_class = net_op(net,x.to(device),threshold)
        y_postop = model.predict(y_op_class[:,1:].to('cpu').numpy())
        y_postop = get_postop(y_postop,y_op).to('cuda')
        iou += iou_score(y_postop, y,num_classes).detach()
        len_dl += x.shape[0]

    return  iou/len_dl

def iou_loss_dl(net, dl,device,num_classes):

    loss = 0   
    iou = 0
    len_dl = 0
    
    for x,y in dl:
        x,y = x.to(device),y.to(device)
        y_op= net(x).detach()
        iou += iou_score(y_op, y,num_classes).detach()
        len_dl += x.shape[0]

    return  iou/len_dl

def confusion_matrix(net,test_dl,model,device): 
    y_op_cm_list = torch.zeros((1,512,512)).to(device)
    y_postop_cm_list = torch.zeros((1,512,512)).to(device)

    for x,y in test_dl:
        x,y = x.to(device),y.to(device)
        y_op,y_op_class = net_op(net,x.to(device),threshold)

        y_postop = model.predict(y_op_class[:,1:].to('cpu').numpy())
        y_postop= clean_op(y_postop,y_op).to(device)
        y_postop_confusion_matrix = y_postop * 1000 + y

        y_op =  torch.argmax(y_op,axis = 1)
        y_op_confusion_matrix = y_op * 1000 + y
        y_postop_cm_list = torch.cat([y_postop_cm_list,y_postop_confusion_matrix],dim = 0)
        y_op_cm_list =  torch.cat([y_op_cm_list,y_op_confusion_matrix],dim = 0)


    y_postop_cm_list = y_postop_cm_list[1:,:,:]
    y_op_cm_list = y_op_cm_list[1:,:,:]   

    postop_cm = pd.DataFrame(zip(*torch.unique(torch.flatten(y_postop_cm_list.to('cpu'), start_dim=1),return_counts = True)),columns = ['label','count_postop']).applymap(lambda x : x.item())
    op_cm = pd.DataFrame(zip(*torch.unique(torch.flatten(y_op_cm_list.to('cpu'), start_dim=1),return_counts = True)),columns = ['label','count_op']).applymap(lambda x : x.item())
    postop_cm.set_index('label',inplace = True)
    op_cm.set_index('label',inplace = True)
    cm = pd.concat([postop_cm,op_cm],axis = 1)
    cm.reset_index('label',inplace = True)
    cm.label = cm.label + 1000

    cm['predicted_class'] = cm.label.apply(lambda x : label_name[int(str(x)[0]) - 1])
    cm['true_class'] = cm.label.apply(lambda x : label_name[int(str(x)[3])])


    cm.set_index(['predicted_class','true_class'],inplace = True)
    cm_postop = cm[['count_postop']].reset_index().pivot(index='predicted_class', columns='true_class', values='count_postop')
    cm_op = cm[['count_op']].reset_index().pivot(index='predicted_class', columns='true_class', values='count_op')
    cm_postop.fillna(0,inplace = True)
    cm_op.fillna(0,inplace = True)

    cm_op_row = cm_op.div(cm_op.sum(axis=1), axis=0)*100
    cm_postop_row =  cm_postop.div(cm_postop.sum(axis=1), axis=0)*100

    cm_op_col = cm_op.div(cm_op.sum(axis=0), axis=1)*100
    cm_postop_col =  cm_postop.div(cm_postop.sum(axis=0), axis=1)*100 
    
    return cm_op_row,cm_postop_row,cm_op_col,cm_postop_col