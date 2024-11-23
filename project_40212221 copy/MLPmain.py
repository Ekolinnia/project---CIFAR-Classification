# Load and normalize the CIFAR10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
from NaiveBayes import NaiveBayes
from DecisionTree import Node
from DecisionTree import DecisionTree
from MLP import MLP
from MLP_lessLayer import MLP_lessLayer
from CNN import CNN
#importing the sklearn library
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import torch.nn as nn
import torch.optim as opti
import torch.nn.functional as F


#for confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#source: https://www.w3schools.com/python/python_ml_confusion_matrix.asp


## ------------------------------------------------------------------------------------------ ## 

# MLP 

#transformation to normalize and flatten the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #Flatten the image
    transforms.Lambda(lambda x: x.view(-1)) 
])



# Import the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Index of each image in list
train_idx = []
test_idx = []

# Each class will have 100 test images and 500 training images
num_train_per_class = 500
num_test_per_class = 100

# Initializing counters for each class (0-9) = 0
train_class = {i: 0 for i in range(10)}
test_class = {i: 0 for i in range(10)}

# First 500 images per class for training
for i, (_, label) in enumerate(trainset):
    if train_class[label] < num_train_per_class:
        # Add to the train_idx list
        train_idx.append(i)
        train_class[label] += 1
    # Stop if all classes have 500 training samples
    if all(count >= num_train_per_class for count in train_class.values()):
        break

# First 100 images per class for testing
for i, (_, label) in enumerate(testset):
    if test_class[label] < num_test_per_class:
        # Add to the test_idx list
        test_idx.append(i)
        test_class[label] += 1
    # Stop if all classes have 100 test samples
    if all(count >= num_test_per_class for count in test_class.values()):
        break

# Subsets of 500 and 100 train and test set per class
subset_trainset = Subset(trainset, train_idx)
subset_testset = Subset(testset, test_idx)

# Data loaders
batch_size = 4
trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(subset_testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Printing the number of train and test set images
print(f'Training data: {len(subset_trainset)} images')
print(f'Testing data: {len(subset_testset)} images')


# Convert subsets to numpy arrays
# Flattened training features (pixels) of images
train_X = np.array([data[0].numpy() for data in subset_trainset])
train_y = np.array([data[1] for data in subset_trainset])

# Convert to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)

# Flattened testing features (pixels) of images
test_X = np.array([data[0].numpy() for data in subset_testset])
test_y = np.array([data[1] for data in subset_testset])

## ------------------------------------------------------------------------------------------ ## 

# MLP 

modelMLP = MLP()


# Set criterion of model to ensure the error and how far the prediction are

criterion = nn.CrossEntropyLoss() 

#SGD optimizer, lower lr longer takes to train

opt = torch.optim.SGD(modelMLP.parameters(), lr = 0.01, momentum=0.9)

# Train model
# Epochs 

epochs = 100
losses = []

for i in range(epochs):
    # go forward and get a prediction
    
    y_predict_mlp = modelMLP.forward(train_X)
    
    # measure loss
    loss = criterion(y_predict_mlp, train_y)
    
    # keep track of losses
    
    losses.append(loss.detach().numpy())
    
    # print loss every 10 epoch
    if i % 10 ==0:
        print(f'Epoch: {i} and loss: {loss}')
    
    # Take error rate of forward propagation and and feed it back in network    
    opt.zero_grad()
    loss.backward()
    opt.step()
      
    
# MLP with lower layers
print(f'MLP with less layer')
    
# model with fewer layers
modelMLP1 = MLP_lessLayer()

# Set criterion of model to ensure the error and how far the prediction are
criterion1 = nn.CrossEntropyLoss()
#SGD optimizer, lower lr longer takes to train

opt1 = torch.optim.SGD(modelMLP1.parameters(), lr=0.01, momentum=0.9)

# Train model
# Epochs 
epochs1 = 100
losses1 = []

for i in range(epochs1):
    # go forward and get a prediction
    y_predict_mlp1 = modelMLP1.forward(train_X)  
    
    # measure loss
    loss1 = criterion1(y_predict_mlp1, train_y)  

    # keep track of losses
    losses1.append(loss1.detach().numpy())

    # print loss every 10 epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss1.item()}')

    # Take error rate of forward propagation and and feed it back in network  
    opt1.zero_grad()     
    loss1.backward()      
    opt1.step()      
         

# predictions for the test set for MLP model
with torch.no_grad():  
    y_predict_mlp = modelMLP(torch.tensor(test_X, dtype=torch.float32))
    y_predict_mlp1 = modelMLP1(torch.tensor(test_X, dtype=torch.float32))

# Convert predictions to class labels using argmax
y_predict_mlp = torch.argmax(y_predict_mlp, dim=1).numpy()
y_predict_mlp1 = torch.argmax(y_predict_mlp1, dim=1).numpy()

# Generate confusion matrices for both MLP
conf_matrix_mlp = confusion_matrix(test_y, y_predict_mlp)
conf_matrix_mlp1 = confusion_matrix(test_y, y_predict_mlp1)

# Display plots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# original MLP model
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
axes[0].set_title("Confusion Matrix for Original MLP")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# MLP with fewer layer
sns.heatmap(conf_matrix_mlp1, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[1])
axes[1].set_title("Confusion Matrix for MLP with Fewer Layers")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


