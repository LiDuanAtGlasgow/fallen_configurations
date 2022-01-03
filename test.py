#type:ignore
from __future__ import print_function
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import os
import time

torch.manual_seed(1)

class Get_Images():
    def __init__(self,image,transforms=None):
        self.image=image
        self.transform=transforms
    
    def __getitem__(self):
        image=self.image
        if not self.transform == None:
            image=self.transform(image)
        image=torch.unsqueeze(image,dim=0)
        return image
    
    def __len__(self):
        return len(self.image)

class KCNet(nn.Module):
    def __init__(self):
        super(KCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*126*126, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(kcnet,data):
    output=kcnet(data)
    print ('output:',output)
    pred=output.argmax(dim=1,keepdim=True)
    print('predicted_label:',pred.item())
    return pred

images_add='./test_images/pos_test_0001/depth/0001.png'
images=cv2.imread(images_add,0)
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.Normalize((0.01183898,), (0.05419697,))
        ])
data=Get_Images(image=images,transforms=transform).__getitem__()
kcnet=KCNet()
kcnet.load_state_dict(torch.load('./Model/KCNet.pt'))
kcnet.eval()
pred=test(kcnet,data)

