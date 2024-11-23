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
#train_X = torch.tensor(train_X, dtype=torch.float32)
#train_y = torch.tensor(train_y, dtype=torch.long)

# Flattened testing features (pixels) of images
test_X = np.array([data[0].numpy() for data in subset_testset])
test_y = np.array([data[1] for data in subset_testset])

#This main file is used for both Naive Bayes and Decision Tree comment out the section u don't 
#want to evaluate

## ------------------------------------------------------------------------------------------ ## 

# Naive Bayes 

#nb = NaiveBayes()
#nb.training(train_X, train_y)
#predictions = nb.predictInput(test_X)
#accuracyOfPredictionNB = np.mean(predictions == test_y) * 100
#print(f"Accuracy with implemented NaiveBayes: {accuracyOfPredictionNB:.2f}%")

# Predict on test set and accuracy from sklearn

#model = GaussianNB()
#model.fit(train_X, train_y)
#predicted_NB = model.predict(test_X)
#accuracyOfPredictionSklearn = np.mean(predicted_NB == test_y) * 100
#print(f"Accuracy with sklearn GaussianNB: {accuracyOfPredictionSklearn:.2f}%")


#building confusion matrix for NB implemented

# Generate confusion matrices for implemented and Sklearn NB Gaussian
#conf_matrix1 = confusion_matrix(test_y, predictions)
#conf_matrix2 = confusion_matrix(test_y, predicted_NB)

#fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot the implemented confusion matrix
#sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
#axes[0].set_xlabel('Predicted Label')
#axes[0].set_ylabel('True Label')
#axes[0].set_title('Confusion Matrix for Custom Naive Bayes')

# Plot the sklearn confusion matrix
#sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[1])
#axes[1].set_xlabel('Predicted Label')
#axes[1].set_ylabel('True Label')
#axes[1].set_title('Confusion Matrix for Sklearn Naive Bayes')

# Display the plots side by side 
#plt.tight_layout()
#plt.show()

## ------------------------------------------------------------------------------------------ ## 


#descision tree

# Predict on test set and accuracy from custom DTC

clf1 = DecisionTree()

clf1.training(train_X, train_y)

predictionDTC = clf1.predict(test_X)

accDTC = accuracy_score(test_y, predictionDTC)*100
print(f"Accuracy with implemented Decision Tree Classifier: {accDTC:.2f}%")

#predict accuracy
accuracyOfPredictionDT = (np.sum(test_y == predictionDTC)/ len(test_y))*100
print(f"Accuracy: {accuracyOfPredictionDT:.2f}%")

# Predict on test set and accuracy from sklearn DTC
clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_X, train_y)
DTCprediction = clf.predict(test_X)
accDTCSklearn = accuracy_score(test_y, DTCprediction)*100
print(f"Accuracy with sklearn Decision Tree Classifier: {accDTCSklearn:.2f}%")


# Generate confusion matrices for implemented and Sklearn DTC
fig_dtc, axes_dtc = plt.subplots(1, 2, figsize=(16, 8))
conf_matrix3 = confusion_matrix(test_y, predictionDTC)
conf_matrix4 = confusion_matrix(test_y, DTCprediction)

# Plot the implemented confusion matrix
sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes_dtc[0])
axes_dtc[0].set_xlabel('Predicted Label')
axes_dtc[0].set_ylabel('True Label')
axes_dtc[0].set_title('Confusion Matrix for Custom Decision Tree Classifier')

# Plot the sklearn confusion matrix
sns.heatmap(conf_matrix4, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes_dtc[1])
axes_dtc[1].set_xlabel('Predicted Label')
axes_dtc[1].set_ylabel('True Label')
axes_dtc[1].set_title('Confusion Matrix for Sklearn Decision Tree Classifier')

# Display the plots side by side 
plt.tight_layout()
plt.show()

## ------------------------------------------------------------------------------------------ ## 

