from glob import glob
import csv
import ipdb
import numpy as np
import pandas as pd
csv_list = glob("./*/*.csv")

dict = {}

for idx,csv_file in enumerate(csv_list):
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if idx==0:
                dict[row[0]]=[]
            dict[row[0]].append(float(row[1]))
subm  = []

for i in dict:
    assert len(dict[i])==20 

    result = np.mean(dict[i])
    ### change the average result which <0 to 0.
    if result<0 or result>100:
        result = 0.0
    subm.append([i,np.mean(dict[i])])
## save same as xjc done.
subm = pd.DataFrame(subm)
subm.to_csv('../Testing_COVIDICIAP_Captain-CSgroup.csv', index = None, header = None)
# import ipdb; ipdb.set_trace()
