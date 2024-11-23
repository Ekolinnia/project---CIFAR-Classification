#source: https://www.youtube.com/watch?v=pDdP0TFzsoQ
import torch
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
import numpy as np

#CNN wITH LesS LAYER

class CNN_lessLayer(nn.Module):
    
    
    def __init__(self, in_channel = 3,num_classes = 10):
        super().__init__()
        #Conv layer and batch
        #input chanel size = 3 cuz 3 RGB, kernel size out channel = 64, stride = 1, padding = 1
        self.conv1 = nn.Conv2d(in_channel,64,3,1,1)
        self.b1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.b2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.b3 = nn.BatchNorm2d(256)
        
        
        # Function connected layer
        self.fc1 = nn.Linear(256*8*8,512)
        self.out = nn.Linear(512,num_classes)
        
        # Pooling layer kernel size = 2, stride = 2
        self.pool = nn.MaxPool2d(2,2)
        
        # Drop out Layer 
        self.dropout = nn.Dropout(0.5)
        
        #function that moves forward in the layers shown project description)  
    def forward(self,x):
        
        #1st layer input
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #2nd layer
        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #3rd layer
        x = self.conv3(x)
        x = self.b3(x)
        x = F.relu(x)
        
        
        # flatten the tensor for the connected layers
        x = torch.flatten(x, 1)
        
        #fully connected layers 
         
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        #output layer
        x = self.out(x)
        
        return x  
        
        

    
    

