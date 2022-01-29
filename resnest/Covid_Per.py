from numpy.lib.shape_base import split
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import PIL 
import os
import os.path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import numpy
import numpy as np
import cv2
import torch.nn.functional as F
import torch.multiprocessing
from PIL import Image, ImageCms
torch.multiprocessing.set_sharing_strategy('file_system')


class Covid_Per(MNIST):
    def __init__(self, root, train, dataset_type, val_index, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform,target_transform=target_transform)
        self.train = train
        self.type = dataset_type
        self.val_index = val_index
        self.data, self.targets, self.sub = torch.load(os.path.join(self.root, self.train))
        x = np.arange(len(self.data)) 
        self.index = numpy.array_split(x, 5)
        self.index_val = self.index[self.val_index]
  
        y = np.delete(self.index,self.val_index,axis = 0)

        self.index_train = numpy.concatenate((y[0],y[1],y[2],y[3]), axis = 0)

        x = np.arange(5)  

    def __getitem__(self, index):

        if self.type == 'train':

            img, target, sub = self.data[self.index_train[index]], self.targets[self.index_train[index]], self.sub[self.index_train[index]]
            img = np.array(img)
            img = img.astype(np.uint8)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return  img, target, sub

        if self.type == 'val':

            img, target, sub = self.data[self.index_val[index]], self.targets[self.index_val[index]], self.sub[self.index_val[index]]
            img = np.array(img)
            img = img.astype(np.uint8)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return  img, target, sub


    def __len__(self):
        if self.type == 'train':
            return int(len(self.data)*0.8)
        if self.type == 'val':
            return int(len(self.data)*0.2)


class Covid_Per_test(MNIST):
    def __init__(self, root, transform=None):
        super(MNIST, self).__init__(root, transform=transform)

        self.imgs = list(sorted(os.listdir(os.path.join(root, "Test"))))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img_path = os.path.join(self.root, "Test", self.imgs[index])
        name = self.imgs[index]
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = img.astype(np.uint8)
      
        if self.transform is not None:
            img = self.transform(img)

            
        return img, name

    def __len__(self):
        return len(self.imgs)


