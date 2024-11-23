import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from CNN import CNN  
from CNN_lessLayer import CNN_lessLayer


#for confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#source: https://www.w3schools.com/python/python_ml_confusion_matrix.asp

# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#transformation CIFAR-10 dataset no need flatten like the other implementations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Import the CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Index of each image in list
train_idx = []
test_idx = []

# Each class will have 100 test images and 500 training images
num_train_per_class = 500
num_test_per_class = 100

# Initializing counters for each class (0-9) = 0
train_class = {i: 0 for i in range(10)}  # Track count per class for training
test_class = {i: 0 for i in range(10)}   # Track count per class for testing

# First 500 images per class for training
for idx, (_, label) in enumerate(trainset):
    if train_class[label] < num_train_per_class:
        # Add to the train_idx list
        train_idx.append(idx)
        train_class[label] += 1
    # Stop if all classes have 500 training samples
    if all(count >= 500 for count in train_class.values()):
        break

    # First 100 images per class for testing
    for idx, (_, label) in enumerate(testset):
        if test_class[label] < num_test_per_class:
            # Add to the test_idx list
            test_idx.append(idx)
            test_class[label] += 1
        # Stop if all classes have 100 test samples    
        if all(count >= 100 for count in test_class.values()):
            break

#Subsets of 500 and 100 train and test set per class
subset_trainset = Subset(trainset, train_idx)
subset_testset = Subset(testset, test_idx)

# Data loaders
# Use bigger Batch size to make training faster...
batch_size = 256
trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(subset_testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
# Print the number of images in each subset to verify
print(f"Training data: {len(subset_trainset)} images")
print(f"Testing data: {len(subset_testset)} images")

## ------------------------------------------------------------------------------------------ ## 

# CNN
# Initialize the model, loss function, and optimizer
# Origina CNN, if want to try with less layer put this below in commment and use the following instanciation
modelCNN = CNN(in_channel=3, num_classes=10).to(device)
#modelCNN = CNN_lessLayer(in_channel=3, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelCNN.parameters(), lr=0.0001, momentum=0.9)
    
# Training model
print("Trainning of CNN model")
# loop for 100 epoch
num_epochs = 10
for epoch in range(num_epochs):
    modelCNN.train()
    running_loss = 0.0
    for i, (imgs, lbls) in enumerate(trainloader):
        imgs, lbls = imgs.to(device), lbls.to(device)
            
        # Forward pass
        outputs = modelCNN(imgs)
        #measure loss
        loss = criterion(outputs, lbls)   
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
            
        # Print progress every 10 batch
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")
        
    # Print loss for each epoch (will have 10)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")
    
# evaluate the test set
Nlabels = []
Npredictions = []
modelCNN.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, lbls in testloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = modelCNN(imgs)
        _, predicted = torch.max(outputs, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()
        
        #store for consuion matrix
        Nlabels.extend(lbls.cpu().numpy())
        Npredictions.extend(predicted.cpu().numpy())
    
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# PSetting the confusion matrix

conf_matrix = confusion_matrix(Nlabels, Npredictions)

 # Ploting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of CNN')
plt.show()
