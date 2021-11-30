
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

class ImageDataset(Dataset):
    ''' The class is used by dataloader to generate tensors of a given batch size.'''
    def __init__(self,final_size,file_location,label_location,image_location,sample_per_class = 100,class_tbl=None):
        
        self.file_location = file_location
        self.height,self.width = final_size
        self.image_location = image_location
        self.sample_per_class = sample_per_class
        self.metadata = pd.read_csv(os.path.join(file_location))
        
        self.ImageList = self.metadata[['ImageId']].drop_duplicates()
        self.ImageList['idx'] = np.array(range(0,len(self.ImageList)))
        self.ImageList.set_index(['idx'],inplace = True)
        self.metadata = self.metadata.merge(self.ImageList.reset_index(),how = 'inner',left_on = 'ImageId',right_on = 'ImageId')
        
        if class_tbl is None:
            f = open(os.path.join(label_location,'label_descriptions.json'))

            data = json.load(f)

            self.class_tbl = pd.DataFrame(data['categories'])
            self.class_len = len(self.class_tbl)
        else:
            self.class_tbl = class_tbl
            self.class_len = len(self.class_tbl)

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, idx):
        '''
        Parameters:
        -------------
        idx : int, Index of the image from the domain determined by the csv file
        
        Returns:
        -------------
        image : floatTensor, Scaled image for the given idx
        op: intTensor, 2D tensor with each pixel labeled by the class
        '''
        
        ImageId = self.ImageList.ImageId[idx]
        metadata = self.metadata[self.metadata.ImageId.isin([ImageId])].drop_duplicates()
        op = torch.zeros(self.height,self.width)
        image = Image.open(os.path.join(self.image_location,ImageId+'.jpg')).convert('RGB')
        image = self.fit_image(image)
        image = torch.tensor(image).permute(2,0,1)
        
        for row in metadata.iterrows():
            
            mask = self.mask2img(row[1])
            mask = self.fit_image(mask)
            op+= (torch.tensor(mask).type(torch.float)*(op ==0).type(torch.float))*(row[1]['ClassId']+1)  
        
        return image.type(torch.float),op.type(torch.int64)
        
    def mask2img(self,metadata):
        ''' mask as a 2D matrix is generated from a string of pixels locations codified in the .csv files'''
        
        metadata['encoded_pixel'] = list(map(int, metadata['EncodedPixels'].split(' ')))

        metadata['encoded_pixel'] =  [list(range(metadata['encoded_pixel'][i],metadata['encoded_pixel'][i]+metadata['encoded_pixel'][i+1])) for i in range(0,len(metadata['encoded_pixel']),2)]
        metadata['encoded_pixel'] = np.array([ i  for sublist in metadata['encoded_pixel'] for i in sublist])

        metadata['data'] =  [1]*len(metadata['encoded_pixel'])
        metadata['row'] = metadata['encoded_pixel']%metadata['Height']
        metadata['col'] = metadata['encoded_pixel']//metadata['Height']
        mask = csr_matrix((metadata['data'], (metadata['row'], metadata['col'])), shape=(metadata['Height'],metadata['Width'])).toarray()
        
        mask =  Image.fromarray(mask.astype(np.uint8))
        return mask
    
    def fit_image(self,image):
        '''The image is scaled to fixed dimensions, it maintains aspect ratio and adds padding to the smaller of the two dimensions'''
        image = self.fit_to_size(image)
        image = self.pad_image(image)
        return np.array(image)
    
    def fit_to_size(self,image):
        '''Scales image while maintaining aspect ratio'''
        #downsize image and mask
        height,width = image.size
        
        width_fit_height = height*self.width/width
        width_fit_width = self.width
        
        height_fit_height = self.height
        height_fit_width = width*self.height/height
        
        if width_fit_height < self.height:
            image = image.resize((int(width_fit_height),self.width), Image.NEAREST)
        else: 
            image = image.resize((self.height,int(height_fit_width)),Image.NEAREST)
            
        return image
    
    
    def upsize(self,image):
        #downsize image and mask
        height,width = image.size
        
        if self.height > height and self.width > width:
            if self.height > height:
                aspect_ratio = width/height
                width = int(aspect_ratio*self.height)
                height = self.height
                image = image.resize((self.height, self.width), Image.NEAREST)

            elif self.width > width:
                aspect_ratio = height/width
                height = int(aspect_ratio*self.width)
                width = self.width
                image = image.resize((self.height, self.width), Image.NEAREST)
            
        return image
    
    def pad_image(self,image):
        '''Pads image across dimension with lower length'''

        height,width = image.size
        self.image = image
        result = Image.new(image.mode, (self.height,self.width))
        offset = ((self.height - height) // 2, (self.width - width) // 2)
        result.paste(image,offset)
        return result
