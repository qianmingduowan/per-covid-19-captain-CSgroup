# Per-Covid-19 Project
# Bougourzi Fares
from torch._C import Value
from Covid_Per import Covid_Per
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import os
import cv2
import pdb
import argparse
from tensorboardX import SummaryWriter
from swintransformer import SwinTransformer
import torchvision.transforms as T
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# import sys
# sys.path.append('/data/xjc/XJC/Per-Covid-19')

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)    

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
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
        # pdb.set_trace()
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

# Use weights from LDS 
def weighted_huber_loss(input, target, beta, weights):
    """
    Dynamic Huber loss function with decreasing 
    beta parameter during training progress
    """
    n = torch.abs(input - target)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)
    if weights is not None:
        loss *= weights.expand_as(loss)
    return loss.mean()

# change from l1 loss to l2 loss
def l1l2_loss(input, target, alpha):
    if alpha>1:
        alpha = 1
    n = torch.abs(input - target)
    loss = alpha * 0.5 * n ** 2 + (1-alpha) * n
    return loss.mean()

# use the best seed
def set_random_seed(seed):
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

# to freeze the first two parts of SwinTransformer
# def freeze(model):
#     for param in model.layers[0].parameters():
#         param.requires_grad = False
#     for param in model.layers[1].parameters():
#         param.requires_grad = False
#     model.layers[0].eval()
#     model.layers[0].eval()

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--validation_part", type=str)
    return parser

if __name__ == "__main__":
    set_random_seed(3407)# set the best seed

    parser = config_parser()
    args = parser.parse_args()
    if args.validation_part == 'fold1':
        train_data = 'train_fold1.pt'
        val_data = 'test_fold1.pt'
        save_dir = 'fold1_model'
    if args.validation_part == 'fold2':
        train_data = 'train_fold2.pt'
        val_data = 'test_fold2.pt'
        save_dir = 'fold2_model'
    if args.validation_part == 'fold3':
        train_data = 'train_fold3.pt'
        val_data = 'test_fold3.pt'
        save_dir = 'fold3_model'
    if args.validation_part == 'fold4':
        train_data = 'train_fold4.pt'
        val_data = 'test_fold4.pt'
        save_dir = 'fold4_model'
    if args.validation_part == 'fold5':
        train_data = 'train_fold5.pt'
        val_data = 'test_fold5.pt'
        save_dir = 'fold5_model'

    print('training ',args.validation_part)

    train_transform = T.Compose([
            T.ToPILImage(mode='RGB'),
            T.RandomChoice([
                T.Compose([
                    T.CenterCrop((450,450)),
                    T.RandomCrop((384,384)),
                    T.Resize((224, 224))
                ]),
                T.Compose([
                    T.CenterCrop((384,384)),
                    T.Resize((224, 224))
                ])]),
            T.RandomApply([T.RandomRotation(degrees = (-10,10))],p=0.5),#旋转角度加到45，加上flip
            # T.RandomVerticalFlip(),
            # T.RandomHorizontalFlip(),
            # T.RandomApply([T.ColorJitter(brightness=0.5, saturation=0.5, hue=0.5)],p=0.5),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])    

    test_transform = T.Compose([
            T.ToPILImage(mode='RGB'),
            T.CenterCrop((384,384)),
            T.Resize((224,224)),
            T.ToTensor()
    ]) 

    print('dataset is loading……')

    train_set = Covid_Per(
            root='./'
            ,train = train_data
            ,transform = train_transform
    )            
    test_set = Covid_Per(
            root='./'
            ,train = val_data
            ,transform = test_transform
    ) 

    device = torch.device("cuda:0") 

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True, num_workers=4)#默认batch_size为20       
    validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 20, num_workers=4)

    print('dataset has been loaded')

    # model = torchvision.models.densenet161(pretrained=True)
    # model.classifier = nn.Linear(2208, 1)
    # model = model.to(device)  

    model = SwinTransformer(img_size = 224,embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7, num_classes=21841)#换用其他的模型试一下
    # model = SwinTransformer()
    model.to(device)
    checkpoint = torch.load('swin_large_patch4_window7_224_22k.pth')
    # model_dict = model.state_dict()
    # pretrain_dict = {k.split('encoder.')[0]:v for k, v in checkpoint['model'].items() if k.split('encoder.')[0] in model_dict}
    # model_dict.update(pretrain_dict)
    model.load_state_dict(checkpoint['model'])
    model.head = nn.Linear(model.num_features, 1)
    model.to(device)

    # model = SwinTransformer(img_size=224, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size = 7, num_classes=21841)#X_ray预训练
    # model.head = nn.Linear(model.num_features, 5)
    # model.to(device)#初始化模型
    # checkpoint = torch.load('swin_large_patch4_window7_224_X_ray.pt')
    # model.load_state_dict(checkpoint)#加载预训练模型
    # model.head = nn.Linear(model.num_features, 1)#将头部切换为回归用途
    # # freeze(model)
    # model.to(device)

    writer = SummaryWriter('./tensorboard')#使用tensorboard
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    
    EPOCH = 90

    # criterion = nn.MSELoss()
    criterion = huber_loss#试一下不同的Loss
    # criterion = weighted_huber_loss
    # criterion = l1l2_loss

    sigma = 2
    beta_max = 15
    beta_min = 1
    pc_best = -2

    lr = 0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr = 0.0003,weight_decay=0.05)

    lr_scheduler = WarmupPolyLR(optimizer,
                                max_iters=len(train_loader)*EPOCH,
                                power=0.9,
                                warmup_iters=len(train_loader)*20)

    for epoch in range(EPOCH):
            
        beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ EPOCH)))
        # beta = 8

        # if epoch == 20:
        #     freeze(model)
        #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, momentum=0.9)
        
        train_loss = np.array([])
        train_sub = np.array([])
        train_pred = np.array([])
        train_label = np.array([])
        test_loss = np.array([])
        test_sub = np.array([])
        test_pred = np.array([])
        test_label = np.array([])
        learning_rate = optimizer.param_groups[0]['lr']

        model.train()
        # if epoch>=20:
        #     model.layers[0].eval()
        #     model.layers[0].eval()
        
        for batch in tqdm(train_loader):                            #在train集上面得出指标
            
            images, labels, weights = batch
            weights = weights.to(device)
            images = images.float().to(device)
            labels = labels.float().to(device)
            preds = model(images)
            # preds = torch.clamp(preds, 0, 1)
            torch.set_grad_enabled(True)
            a = images[0].cpu().numpy().transpose(1, 2, 0)
            cv2.imwrite('img.png',a*255)    

            loss = criterion(preds.squeeze(1), labels, beta)  
            # loss = criterion(preds.squeeze(1), labels, epoch/(EPOCH-10))   

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss=np.append(train_loss,loss.cpu().detach().numpy())
            train_sub=np.append(train_sub,weights.cpu().detach().numpy())
            train_pred=np.append(train_pred,preds.cpu().detach().numpy())
            train_label=np.append(train_label,labels.cpu().detach().numpy())  

            lr_scheduler.step()

        model.eval()                
        for batch in tqdm(validate_loader):                                  #在test集上面得出指标
            
            images, labels, weights = batch
            images = images.float().to(device)
            labels = labels.float().to(device)
            weights = weights.to(device)

            with torch.no_grad():
                preds = model(images)
                loss = criterion(preds.squeeze(1), labels,beta)  
                # loss = criterion(preds.squeeze(1), labels, epoch/(EPOCH-10))
                #              
            test_loss=np.append(test_loss,loss.cpu().detach().numpy())
            test_sub=np.append(test_sub,weights.cpu().detach().numpy())
            test_pred=np.append(test_pred,preds.cpu().detach().numpy())
            test_label=np.append(test_label,labels.cpu().detach().numpy())  
            

        train_MAE = np.mean(np.abs(train_pred-train_label))
        test_MAE = np.mean(np.abs(test_pred-test_label))
        train_RMSE = np.mean((train_pred-train_label)**2)
        test_RMSE = np.mean((test_pred-test_label)**2)
        train_PC = PC_mine(train_pred, train_label)
        test_PC = PC_mine(test_pred, test_label)

        PC_dict = {'train':train_PC, "val":test_PC}
        MAE_dict = {'train':train_MAE, "val":test_MAE}
        loss_dict = {'train':np.mean(train_loss), 'val':np.mean(test_loss)}
        RMSE_dict = {'train':train_RMSE, 'test':test_RMSE}
        

        writer.add_scalars('PC', PC_dict, epoch)
        writer.add_scalars('MAE', MAE_dict, epoch)
        writer.add_scalars('Loss', loss_dict, epoch)
        writer.add_scalars('RMSE', RMSE_dict, epoch)
        writer.add_scalar('lr', learning_rate, epoch)

        print(epoch,'------train--------test----')
        print('Loss  %4.8f  %4.8f'%(np.mean(train_loss), np.mean(test_loss)))
        print('MAE   %4.8f  %4.8f'%(train_MAE, test_MAE))
        print('RMSE  %4.8f  %4.8f'%(train_RMSE, test_RMSE))
        print('PC    %4.8f  %4.8f'%(train_PC, test_PC))
        print('lr    ', learning_rate)

        if (epoch==EPOCH-1):
            torch.save(model.state_dict(), './'+save_dir+'/SwinTransformer_'+ str(epoch)+'_epoch.pt')
