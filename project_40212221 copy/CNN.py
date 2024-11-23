#source: https://www.youtube.com/watch?v=pDdP0TFzsoQ
import torch
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    
    
    def __init__(self, in_channel = 3,num_classes = 10):
        super().__init__()
        #Conv layer and batch
        #input chanel size = 3 cuz 3 RGB, kernel size out channel = 64, stride = 1, padding = 1
        #self.conv1 = nn.Conv2d(in_channel,64,3,1,1) #change kernel size for testing 7x7
        self.conv1 = nn.Conv2d(in_channel,64,7,1,3)
        self.b1 = nn.BatchNorm2d(64)
        
        #self.conv2 = nn.Conv2d(64,128,3,1,1) #5x5
        self.conv2 = nn.Conv2d(64,128,5,1,2)
        self.b2 = nn.BatchNorm2d(128)
        
        #self.conv3 = nn.Conv2d(128,256,3,1,1) #5x5
        self.conv3 = nn.Conv2d(128,256,5,1,2) 
        self.b3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256,256,3,1,1)
        self.b4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256,512,3,1,1)
        self.b5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512,512,3,1,1)
        self.b6 = nn.BatchNorm2d(512)
        
        self.conv7 = nn.Conv2d(512,512,3,1,1)
        self.b7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512,512,3,1,1)
        self.b8 = nn.BatchNorm2d(512)
        
        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.b9 = nn.BatchNorm2d(512)
        
        # Function connected layer
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.out = nn.Linear(4096,num_classes)
        
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
        
        #4th layer
        x = self.conv4(x)
        x = self.b4(x)
        x = F.relu(x)
        
        #5th layer
        x = self.conv5(x)
        x = self.b5(x)
        x = F.relu(x)
        
        #6th layer
        x = self.conv6(x)
        x = self.b6(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #7th layer
        x = self.conv7(x)
        x = self.b7(x)
        x = F.relu(x)   
        
        #8th layer  
        x = self.conv8(x)
        x = self.b8(x)
        x = F.relu(x)
        x = self.pool(x)  
        
        # flatten the tensor for the connected layers
        x = torch.flatten(x, 1)
        
        #fully connected layers 
        
        #9th layer 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        #10th layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        #output layer
        x = self.out(x)
        
        return x  
        
        

    
    

