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
    op = net(x)
    op = F.softmax(op)
    op = (op >= threshold).type(torch.float)
    op_area= op.mean(dim = 2).mean(dim = 2)
    return op,op_area

def y_one_hot(y,num_classes):
    y_class =  F.one_hot(y,num_classes = num_classes).permute([0,3,1,2])
    y_class =torch.max(y_class,2).values
    y_class = torch.max(y_class,2).values
    return y_class

def normalize(img):
        
    batch_size = img.shape[0]
    img_view = img.view(batch_size,3,-1)
    img_std,_ = img_view.max(dim = 2)
    img = (img)/img_std[:,:,None,None]

    return img

def clean_op(y_predict,y_op):
    y_predict = np.concatenate([np.ones((y_predict.shape[0],1)),y_predict],axis = 1)
    y_predict = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(y_predict),dim = 2),dim = 2)
    y_predict_op = y_op.to('cpu')*y_predict
    y_predict_op[:,0,:,:] = (y_predict_op[:,1:,:,:].sum(axis = 1) == 0).type(torch.float)
    y_predict_op = torch.argmax(y_predict_op,axis = 1)
    return y_predict_op

def dataloader(image_location,file_name,batch_size,sample_per_class=None):
    
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

