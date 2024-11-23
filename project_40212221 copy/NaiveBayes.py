#NaivesBayes Function
#Source :https://www.youtube.com/watch?v=TLInuAorxqE


import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

class NaiveBayes:

    #where X number of samples and y is the prediction
    def training(self, X, y):
        
        #takes row  (samepl) and colums (feature)
        numSamples, numFeatures = X.shape
        
        #store unique class label of target array y
        self._classes = np.unique(y)
        #get num of class
        numClasses = len(self._classes)

        #calculate mean, var, and prior for each class
        self._mean = np.zeros((numClasses, numFeatures), dtype=np.float64)
        self._var = np.zeros((numClasses, numFeatures), dtype=np.float64)
        self._priors = np.zeros(numClasses, dtype=np.float64)
        
        #itirate over each class stored 
        for idx, c in enumerate(self._classes):
            #current training data of class and select only rows from X that corrstponds to class
            #pixel values for images of a class
            X_class = X[y == c]
            #calculate feature (pixel) across rows
            #average value of pixels
            self._mean[idx, :] = X_class.mean(axis=0)
            #variability of pixels in the class
            self._var[idx, :] = X_class.var(axis=0)
            #Calculate prior possibility of class total sample of class/ over all samples 
            self._priors[idx] = X_class.shape[0] / float(numSamples)
            

    #predic input data x in dataset X
    def predictInput(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self.probDensityFunction(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    #probability density function = prob of each feature given a class
    def probDensityFunction(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(-((x - mean) ** 2) / (2 * var))
        dem = np.sqrt(2 * np.pi * var)
        return num / dem