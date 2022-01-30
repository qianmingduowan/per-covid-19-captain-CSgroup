"""
Created on Fri Apr  2 01:35:03 2021

@author: bougourzi
"""

# Per-Covid-19 Project
# Bougourzi Fares
from Covid_Per import Covid_Per
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import random_split
from pathlib import Path
from tensorboardX import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(linewidth=120)   

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # parser.add_argument("--save_dir", type=str)
    parser.add_argument("--validation_part", type=str)

    return parser

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=1e-10, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

# Dynamic Huber loss
def huber_loss(input, target, beta):
    """
    Dynamic Huber loss function with decreasing 
    beta parameter during training progress
    """
    n = torch.abs(input - target)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()


def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def Adaptive_loss(preds, labels, sigma):
    mse = (1+sigma)*((preds - labels)**2)
    mae = sigma + (torch.abs(preds - labels))
    return torch.mean(mse/mae)

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 



def main(epoch_all,bs=80,tensorboard_path='./tensorboard',output_dir = 'output-resnest50_q'): ## 这里别动 在最下面一行改路径什么的
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((320,320)),
        transforms.RandomRotation(degrees = (-10,10)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    parser = config_parser()
    args = parser.parse_args()

    if args.validation_part == '0':
        train_dataset = Covid_Per(root='../',val_index = 0,dataset_type = 'train',train = 'full_train_fold.pt',transform = train_transform)
        val_dataset = Covid_Per(root='../',val_index = 0,dataset_type = 'val',train = 'full_train_fold.pt',transform = train_transform)
        output_dir = 'output_resnest200_0'
    if args.validation_part == '1':
        train_dataset = Covid_Per(root='../',val_index = 1,dataset_type = 'train',train = 'full_train_fold.pt',transform = train_transform)
        val_dataset = Covid_Per(root='../',val_index = 1,dataset_type = 'val',train = 'full_train_fold.pt',transform = train_transform)
        output_dir = 'output_resnest200_1'
    if args.validation_part == '2':
        train_dataset = Covid_Per(root='../',val_index = 2,dataset_type = 'train',train = 'full_train_fold.pt',transform = train_transform)
        val_dataset = Covid_Per(root='../',val_index = 2,dataset_type = 'val',train = 'full_train_fold.pt',transform = train_transform)
        output_dir = 'output_resnest200_2'
    if args.validation_part == '3':
        train_dataset = Covid_Per(root='../',val_index = 3,dataset_type = 'train',train = 'full_train_fold.pt',transform = train_transform)
        val_dataset = Covid_Per(root='../',val_index = 3,dataset_type = 'val',train = 'full_train_fold.pt',transform = train_transform)
        output_dir = 'output_resnest200_3'
    if args.validation_part == '4':
        train_dataset = Covid_Per(root='../',val_index = 4,dataset_type = 'train',train = 'full_train_fold.pt',transform = train_transform)
        val_dataset = Covid_Per(root='../',val_index = 4,dataset_type = 'val',train = 'full_train_fold.pt',transform = train_transform)
        output_dir = 'output_resnest200_4'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    tensorboard_path = os.path.join(output_dir,tensorboard_path)

    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    writer = SummaryWriter(tensorboard_path)

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, num_workers = 4, shuffle = True)
   
    # model
    from resnest.torch import resnest200
    model = resnest200(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model = model.to(device) 

    # criterion
    criterion = huber_loss
    criterion_1 = nn.L1Loss()
    criterion_2 = nn.MSELoss()

    last_epoch = 0
    #load
    model_state_file = os.path.join(output_dir,'checkpoint.pth.tar')

    if os.path.isfile(model_state_file):
        print("load model")
        checkpoint = torch.load(model_state_file)
        last_epoch = checkpoint['epoch']
        dct = checkpoint['state_dict']
        model.load_state_dict(dct)

    sigma = 2
    
    beta_max = 15
    beta_min = 1
    
    pc_best = -2

    total_loss = 0
    total_correct = 0

    tr_sub = []

    train_PC = []

    epoch_count = []
    lr = 0.0003
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    scheduler  = WarmupPolyLR(optimizer,max_iters=len(train_loader)*(epoch_all+1),power=0.9,warmup_iters=len(train_loader)*20)
    model.train()


    for epoch in range(last_epoch,epoch_all+1):
        epoch_count.append(epoch)
        beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ epoch_all+1)))
        alpha = 1-(epoch/(epoch_all-10))*1
        if alpha < 0:
            alpha = 0
        itr = -1
        total_loss = 0
        total_correct = 0
        labels2_tr = np.array([])
        labels_pred_tr = np.array([])

        print("train begin")
        for batch in tqdm(train_loader):  
            itr +=1
            optimizer.zero_grad()
            images, labels, sub = batch
            images = images.float().to(device)
            labels = labels.float().to(device)
            preds = model(images)

            loss_1  = criterion_1(preds.squeeze(),labels.squeeze())
            loss_2 = criterion_2(preds.squeeze(),labels.squeeze())
            blending_loss = alpha*loss_1 + (1-alpha)*loss_2

            torch.nn.utils.clip_grad_norm_(model.parameters(), 15.0)
            total_loss += blending_loss.item()            
            total_correct += MAE_distance(preds.squeeze(1), labels)

            tr_sub.append(sub)
            blending_loss.backward()
            optimizer.step()            
            labels2_tr = np.append(labels2_tr,labels.cpu().detach().numpy())
            labels_pred_tr = np.append(labels_pred_tr,preds.cpu().detach().numpy())
            scheduler.step()


        writer.add_scalar('blending_loss', total_loss/len(train_loader), epoch)
        writer.add_scalar('alpha', alpha, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch) 
        writer.add_scalar('MAE_tr', total_correct/len(train_dataset), epoch)       
        train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
       
        print('Ep: ', epoch,  'MAE_tr: ', total_correct/len(train_dataset), 'loss_tr:', total_loss/len(train_loader),'PC_tr:',PC_mine(labels_pred_tr, labels2_tr))
        
        #save
        print("save")
        torch.save({
            'epoch': epoch +1,
            'state_dict': model.state_dict(),
        }, os.path.join(output_dir,'checkpoint.pth.tar'))

        if epoch %10 == 0 or epoch > epoch_all - 10:
            torch.save({
                'epoch': epoch +1,
                'state_dict': model.state_dict(),
            }, os.path.join(output_dir,'checkpoint-%d.pt' % epoch))
                            
    writer.close()

if __name__ == '__main__':
    main(epoch_all = 90,bs=30,tensorboard_path='./tensorboard_q')

