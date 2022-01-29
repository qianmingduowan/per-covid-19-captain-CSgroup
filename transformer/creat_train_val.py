
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
from scipy.ndimage import convolve1d
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import tqdm as tqdm 
import matplotlib.pyplot as plt
import xlrd
import csv
from sklearn.model_selection import train_test_split, KFold

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
 
database_path = "/data/dataset/Convid/Train/"
excel_path = "/data/dataset/Convid/Train.csv"
images_name = []
covid_per = []
sub_name = []
label_num = [0 for i in range(101)]

with open(excel_path, 'r') as f:
    data_csv = csv.reader(f)  
    for i in data_csv:                                      
        images_name.append(i[0])                            
        covid_per.append(float(i[1]))
        label_num[int(float(i[1]))]+=1                            
        sub_name.append(int(i[2]))                            

label_num = label_num/sum(np.array(label_num))
print(label_num)

lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
eff_label_dist = convolve1d(np.array(label_num), weights=lds_kernel_window, mode='constant')
print(eff_label_dist)

label_weight = 1/eff_label_dist
label_weight = label_weight/label_weight[0]
print(label_weight)

while 1:
    pass
img_train = []
label_train = []
weight_train = []
img_test = []
label_test = []
sub_test = []
sub_test = []
for i in range(len(images_name)):
    full_path_image = os.path.join(database_path, images_name[i])
    img = cv2.imread(full_path_image)
    img = np.array(img)
    # if sub_name[i]%6==1:
    #     img_test.append(img)
    #     label_test.append(covid_per[i])
    #     sub_test.append(sub_name[i])
    # else:
    #     img_train.append(img)
    #     label_train.append(covid_per[i])
    #     sub_train.append(sub_name[i])

    # for j in range(label_weight[int(covid_per[i])]):
    #     print(i, covid_per[i], label_weight[int(covid_per[i])])
    #     img_train.append(img)
    #     label_train.append(covid_per[i])
    #     sub_train.append(sub_name[i])

    img_train.append(img)
    label_train.append(covid_per[i])
    weight_train.append(label_weight[int(covid_per[i])])

training= (img_train, label_train, weight_train)
torch.save(training,'full_train_fold_weighted.pt') 

# testing= (img_test, label_test, sub_test)
# torch.save(testing,'test_fold.pt')       
            