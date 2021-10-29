#type:ignore
import os
import cv2
import pandas as pd

target_file='./Database/RGB_GRAYSCALE/'
if not os.path.exists(target_file):
    os.makedirs(target_file)

data_csv=pd.read_csv('./labels.csv')
for i in range(len(data_csv)):
    image=cv2.imread('./Database/RGB/'+data_csv.iloc[i,0])
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_file+data_csv.iloc[i,0],gray)
print ('finished!')
