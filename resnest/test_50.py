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

from  torch.autograd import Variable
import torch.utils.data as Data
import csv
import codecs
from Covid_Per import Covid_Per_test


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # parser.add_argument("--save_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_file", type=str)

    return parser


def main(output_dir,data_file):
    
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    data_file = args.data_file

    
    #prepare data
    test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        # transforms.CenterCrop((400,400)),
        # transforms.Resize((256,256)),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]) 
    
    test_dataset = Covid_Per_test(
        root ='../dataset/Convid/',
        transform = test_transform
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1)

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    from resnest.torch import resnest50
    model = resnest50(pretrained=False)

    model.fc = nn.Linear(2048, 1)

    print("load model")
    model_state_file = os.path.join(output_dir)

    checkpoint = torch.load(model_state_file)
    dct = checkpoint['state_dict']
    model.load_state_dict(dct)
    model = model.to(device) 

    print("make a dir")
    os.makedirs(os.path.join('./', 'prediction'), exist_ok=True)
    data_file = os.path.join('./', 'prediction', data_file)
    f = open(data_file, 'w') 
    csv_writer = csv.writer(f)
    
    print("test begin")
    for batch in tqdm(test_loader):          
        image, name = batch
        image = image.float().to(device)
        model.eval()
        with torch.no_grad():
            pred = model(image)

            pred =pred.squeeze(1)[0].item()

        csv_writer.writerow([name[0],pred])
        
    f.close()
if __name__ == '__main__':
    main(output_dir = 'output_resnest50_4/checkpoint-89.pt', data_file= 'prediction_resnest50_4.csv')
      
    
