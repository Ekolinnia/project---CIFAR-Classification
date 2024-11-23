#source: https://www.youtube.com/watch?v=JHWqWIoac2I
#        https://www.youtube.com/watch?v=Xp0LtPBcos0
import torch
import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F
import numpy as np
import random

class MLP_lessLayer(nn.Module):
    # Input layer (features) H1 (num of neurons) ->  H2 (n) --> output ( 10 classes in datasheet ) 
    # Will have 32pixels * 32pixels * 3RGB = 3072  features
    def __init__(self, input_features = 3072, h1 = 512, output_features = 10 ):
        super().__init__() #instantiate the nn.Module
        self.fc1 = nn.Linear(input_features,h1)
        self.out = nn.Linear(h1,output_features)
    
        
    #function that moves forward in the hidden layers    
    def forward(self,x):
        # Rectified Linear Unit, if more than 0, use the number, less than 0 output will be 0
        # push 1st layer
        x = F.relu(self.fc1(x))
        
        x = self.out(x)
        
        return x  
    
# manual seed for randomization
torch.manual_seed(41)



               
           
