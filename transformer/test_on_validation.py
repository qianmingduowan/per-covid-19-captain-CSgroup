import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import os
from swintransformer import SwinTransformer
import torchvision.transforms as T
import numpy as np
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    return parser

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = SwinTransformer(img_size=224, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size = 7, num_classes=21841)
    # model = SwinTransformer(img_size = 384, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12, num_classes=21841)
    # model = SwinTransformer()  
    model.head = nn.Linear(model.num_features, 1)
    model.to(device)
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)

    # model = torchvision.models.densenet161(pretrained=True)
    # model.classifier = nn.Linear(2208, 1)
    # model = model.to(device) 
    # model.load_state_dict(torch.load('/data/xjc/XJC/Per-Covid-19/Models_/SwinTransformer_65_epoch.pt'))

    model.to(device)
    model.eval()
    transform = T.Compose([
                T.ToPILImage(mode='RGB'),
                T.CenterCrop((384,384)),
                T.Resize((224,224)),
                T.ToTensor()
        ]) 

    root = "../dataset/Convid/Test/"
    filenames = os.listdir(root)

    subm = []
    with torch.no_grad():
        for i in tqdm(range(len(filenames))):
            file = 'Image_'+'%04d'%i+'.png'
            path = os.path.join(root, file)
            img = cv2.imread(path)
            img = np.array(img)
            img = transform(img)
            img = img.unsqueeze(0).cuda()
            pred = model(img)
            subm.append([file, pred.item()])

        
    subm = pd.DataFrame(subm)
    save_path = args.model_dir.split('/')[0]
    print(save_path)
    subm.to_csv(save_path+'.csv', index = None, header = None)
        