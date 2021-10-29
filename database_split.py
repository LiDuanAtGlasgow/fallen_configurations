# type: ignore
import sys
import numpy as np
import os
import cv2
import pandas as pd
import csv
import time

np.random.seed(42)
csv_path='./labels.csv'
files=pd.read_csv(csv_path)

indices=np.zeros(len(files))
for i in range(int(len(files)*0.2)):
    indices[i]=1
indices=np.random.permutation(indices)

f_train=open('./train.csv','w')
f_test=open('./val.csv','w')
train_writer=csv.writer(f_train)
test_writer=csv.writer(f_test)
train_writer.writerow(('name','pos'))
test_writer.writerow(('name','pos'))
print('start!')
start_time=time.time()
for i in range(len(files)):
    if indices[i]==0:
        train_writer.writerow((files.iloc[i,0],files.iloc[i,1]))
    else:
        test_writer.writerow((files.iloc[i,0],files.iloc[i,1]))
    if i%int(len(files)/10)==0:
        print(f'Files:{i}/{len(files)} have been completed, time elapsed:{time.time()-start_time}')
f_train.close()
f_test.close()
print('finished!')
