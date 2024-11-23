import numpy as np
import random
#source: https://www.youtube.com/watch?v=NxEHSAfFlK8&t=271s

class Node:
    def __init__(self, featureIndex, threshold, left, right):
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        
        
class DecisionTree:
    def __init__(self, minSampleSplits=10, maxDepth=100, maxFeatures=None, maxThresholds=10):
        
        self.minSampleSplits = minSampleSplits
        self.maxDepth = maxDepth
        # features each split
        self.maxFeatures = maxFeatures  
        # Since data is big, limit sample per feature
        self.maxThresholds = maxThresholds  
        self.tree = None

    def training(self, X, y):
        # Growing tree 
        self.tree = self.growTree(X, y)

    # Expand the tree left side and the right side
    def growTree(self, X, y, depth=0):
        numSamples, numFeatures = X.shape
        numLabels = len(np.unique(y))

        # help debug
        print(f"depth: {depth}, samples: {numSamples}, unique labels: {numLabels}")

        # Stop when leaf node
        if (depth >= self.maxDepth or numLabels == 1 or numSamples < self.minSampleSplits):
            leafValue = self.common_label(y)
            return leafValue

        # Random sample features for splitting
        if self.maxFeatures is None:
            self.maxFeatures = int(np.sqrt(numFeatures))  # Use sqrt(numFeatures) as default
        features = np.random.choice(numFeatures, self.maxFeatures, replace=False)

        # Finding the best split
        bestSplit = self.getBestSplit(X, y, numSamples, features)
        # Findin common label in the leaf node
        if bestSplit["infoGain"] == 0:
            leafValue = self.common_label(y)
            return leafValue

        
        left_side = self.growTree(bestSplit["Xleft"], bestSplit["yleft"], depth + 1)
        right_side = self.growTree(bestSplit["Xright"], bestSplit["yright"], depth + 1)
        return Node(bestSplit["featureIndex"], bestSplit["threshold"], left_side, right_side)

    #Getting best split with gini index calculation
    def getBestSplit(self, X, y, numSamples, features):
        bestSplit = {}
        max_IG = -float("inf")

        for featureIndex in features:
            
            featureValues = X[:, featureIndex]
            possible_thresholds = np.unique(featureValues)

            # Sample a limited number of thresholds
            if len(possible_thresholds) > self.maxThresholds:
                possible_thresholds = np.random.choice(possible_thresholds, self.maxThresholds, replace=False)

            
            for threshold in possible_thresholds:
                # do split based on the threshold
                Xleft, yleft, Xright, yright = self.split(X, y, featureIndex, threshold)
                if len(yleft) > 0 and len(yright) > 0:
                    # Calculate information gain
                    current_IG = self.informationGain(y, yleft, yright)
                    if current_IG > max_IG:
                
                        bestSplit = {
                            #parameters saved for splitting
                            "featureIndex": featureIndex,
                            "threshold": threshold,
                            "Xleft": Xleft, "yleft": yleft,
                            "Xright": Xright, "yright": yright,
                            "infoGain": current_IG
                        }
                        max_IG = current_IG
        return bestSplit

    def split(self, X, y, featureIndex, threshold):
        leftIndices = np.where(X[:, featureIndex] <= threshold)
        rightIndices = np.where(X[:, featureIndex] > threshold)
        return X[leftIndices], y[leftIndices], X[rightIndices], y[rightIndices]

    def informationGain(self, y, yleft, yright):
        # Calculate Gini index 
        weightLeft = len(yleft) / len(y)
        weightRight = len(yright) / len(y)
        # Information gain 
        gain = self.giniIndex(y) - (weightLeft * self.giniIndex(yleft) + weightRight * self.giniIndex(yright))
        return gain

    def giniIndex(self, y):
        # Calculate Gini index for the labels
        classes, counts = np.unique(y, return_counts=True)
        gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)
        return gini

    def common_label(self, y):
        
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        #make prediction by traversing the tree
        return [self.makePrediction(x, self.tree) for x in X]

    def makePrediction(self, x, tree):
        # Traverse the tree left or right depending on threshold
        if not isinstance(tree, Node):
            return tree
        if x[tree.featureIndex] <= tree.threshold:
            return self.makePrediction(x, tree.left)
        else:
            return self.makePrediction(x, tree.right)

