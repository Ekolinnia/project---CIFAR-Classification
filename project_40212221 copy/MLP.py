#source: https://www.youtube.com/watch?v=JHWqWIoac2I
#        https://www.youtube.com/watch?v=Xp0LtPBcos0
import torch
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
import numpy as np
import random

class MLP(nn.Module):
    # Input layer (features) H1 (num of neurons) ->  H2 (n) --> output ( 10 classes in datasheet ) 
    # Will have 32pixels * 32pixels * 3RGB = 3072  features
    # can increase the with to another for testing h1,h2 = 1024 and in batch norm 512 -> 1024
    def __init__(self, input_features = 3072, h1 = 512, h2 = 512, output_features = 10 ):
        super().__init__() #instantiate the nn.Module
        self.fc1 = nn.Linear(input_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,output_features)
        self.b1 = nn.BatchNorm1d(512)
        
    #function that moves forward in the hidden layers    
    def forward(self,x):
        # Rectified Linear Unit, if more than 0, use the number, less than 0 output will be 0
        # push 1st layer
        x = F.relu(self.fc1(x))
        # push 2nd layer
        x = self.fc2(x)
        x = self.b1(x)
        x = F.relu(x)
        # output layer 
        x = self.out(x)
        
        return x  
    
# manual seed for randomization
torch.manual_seed(41)




               
           
