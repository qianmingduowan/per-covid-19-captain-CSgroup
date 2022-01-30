
import torch
import numpy as np
import math
import cv2
import csv
import os
from skimage import  transform 
import scipy.io as sio
import numpy as np
import cv2
import os
import argparse
import tqdm as tqdm 
import matplotlib.pyplot as plt
import xlrd
import csv
from sklearn.model_selection import train_test_split, KFold

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str)
    parser.add_argument("--excel_path", type=str)
    return parser
parser = config_parser()
args = parser.parse_args()

# database_path = "/data/dataset/Convid/Train/"
# excel_path = "/data/dataset/Convid/Train.csv"
database_path = args.database_path
excel_path = args.excel_path
images_name = []
covid_per = []
sub_name = []
folds = []

with open(excel_path, 'r') as f:
    data_csv = csv.reader(f)  
    for i in data_csv:                                      
        images_name.append(i[0])                            
        covid_per.append(float(i[1]))                            
        sub_name.append(int(i[2]))                            
        #folds.append(data_excel.cell_value(i,2))           
        folds.append(int(i[2])%5+1)    

Training_data = []
Training_label = []
Fold1 = []
Fold2 = []
Fold3 = []
Fold4 = []
Fold5 = []

i = -1
for line in images_name:
    i += 1
    img_name= line
    full_path_image = os.path.join(database_path, img_name)
    img = cv2.imread(full_path_image)

    Training_data.append(np.array(img))
    Training_label.append(float(covid_per[i]))
    if folds[i] == 1:                                    
        Fold1.append(i)
    elif folds[i] == 2:
        Fold2.append(i)
    elif folds[i] == 3:
        Fold3.append(i) 
    elif folds[i] == 4:
        Fold4.append(i)
    elif folds[i] == 5:
        Fold5.append(i)
print('data is ready')
 

################## 1 ###############################
train_indx1 = Fold2 + Fold3 + Fold4 + Fold5  
X_train = [Training_data[i] for i in train_indx1]  
y_train = [Training_label[i] for i in train_indx1]  
sub_train = [sub_name[i] for i in train_indx1] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold1.pt') 

X_test = [Training_data[i] for i in Fold1]  
y_test = [Training_label[i] for i in Fold1] 
sub_test = [sub_name[i] for i in Fold1] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold1.pt')       
            
################### 2 ############################

train_indx2 = Fold1 + Fold3 + Fold4 + Fold5   

X_train = [Training_data[i] for i in train_indx2]  
y_train = [Training_label[i] for i in train_indx2]  
sub_train = [sub_name[i] for i in train_indx2] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold2.pt') 
    

X_test = [Training_data[i] for i in Fold2]  
y_test = [Training_label[i] for i in Fold2] 
sub_test = [sub_name[i] for i in Fold2] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold2.pt') 

################### 3 ###########################
train_indx3 = Fold1 + Fold2 + Fold4 + Fold5   

X_train = [Training_data[i] for i in train_indx3]  
y_train = [Training_label[i] for i in train_indx3]  
sub_train = [sub_name[i] for i in train_indx3] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold3.pt') 
    

X_test = [Training_data[i] for i in Fold3]  
y_test = [Training_label[i] for i in Fold3] 
sub_test = [sub_name[i] for i in Fold3] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold3.pt') 

################## 4 ############################
train_indx4 = Fold1 + Fold2 + Fold3 + Fold5   

X_train = [Training_data[i] for i in train_indx4]  
y_train = [Training_label[i] for i in train_indx4]  
sub_train = [sub_name[i] for i in train_indx4] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold4.pt') 
    

X_test = [Training_data[i] for i in Fold4]  
y_test = [Training_label[i] for i in Fold4] 
sub_test = [sub_name[i] for i in Fold4] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold4.pt') 

#################### 5 ##########################

train_indx5 = Fold1 + Fold2 + Fold3 + Fold4   

X_train = [Training_data[i] for i in train_indx5]  
y_train = [Training_label[i] for i in train_indx5]  
sub_train = [sub_name[i] for i in train_indx5] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold5.pt') 
    

X_test = [Training_data[i] for i in Fold5]  
y_test = [Training_label[i] for i in Fold5] 
sub_test = [sub_name[i] for i in Fold5] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold5.pt') 

##############################################

