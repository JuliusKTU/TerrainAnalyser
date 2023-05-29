from sklearn.datasets import (make_blobs, make_circles, make_moons)
import matplotlib.pyplot as plt
from pandas import DataFrame
import random
import numpy as np
from matplotlib.animation import FuncAnimation

class KMeans_Soft:
    def __init__(self, Xvalues, number_of_clusters):
        self.Xvalues = Xvalues
        self.number_of_clusters = number_of_clusters
        
    def analyse_terrain_image(self):
        np.random.seed(1)
        X, y = make_blobs(n_samples=len(self.Xvalues), centers=3, n_features=2)
        X = self.Xvalues
        X[:,0]=(X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
        X[:,1]=(X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
        colors = {0:'red', 1:'blue', 2:'green',3:'magenta', 4:'blue'}
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y',  color='blue') # label=key,
            
        numberOfClusters = self.number_of_clusters
        sigma = 0.25 # measure distance in units of σ
        fig, ax =  plt.subplots()
        centroids = dict()
        indices = np.random.choice(len(y), size=numberOfClusters, replace=False)
        for i in range(0,numberOfClusters):
            centroids[i]= X[indices[i],0],X[indices[i],1]
        clusterLabel = np.zeros((len(y),numberOfClusters)) # labels (scores for each centroid)

        # distances 
        for k in range(numberOfClusters):
            centerX = centroids[k]
            distTemp = (X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2 # distance to k cluster
            affinity = np.exp(-distTemp/(2*sigma))
            clusterLabel[:,k] = affinity

        # creating dataframe
        df = DataFrame(dict(x=X[:,0], y=X[:,1]))
        weights = clusterLabel/np.sum(clusterLabel, axis = 1, keepdims=True)
        df.plot(ax=ax, kind='scatter', x='x', y='y', color=weights) 

        for i in range(numberOfClusters):
            centroids[i] = [np.sum(weights[:,i]*X[:,0]),np.sum(weights[:,i]*X[:,1])]/np.sum(weights[:,i])

        numberOfIterations = 10 # max number of iteration
        epsilon = 1e-6
        for iteration in range(0,numberOfIterations):

            df = DataFrame(dict(x=X[:,0], y=X[:,1]))
            fig, ax =  plt.subplots()
            for i in range(0, numberOfClusters):
                ax.scatter(x=centroids[i][0], y=centroids[i][1], color='orange', marker='D', s=100)

            clusterLabel = np.zeros((len(y),numberOfClusters))
            for k in range(numberOfClusters):
                centerX = centroids[k]
                distTemp = (X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2 # distance to k cluster
                affinity = np.exp(-distTemp/(2*sigma))
                clusterLabel[:,k] = affinity

            weightsTemp = clusterLabel/np.sum(clusterLabel, axis = 1, keepdims=True)
            df.plot(ax=ax, kind='scatter', x='x', y='y', color=weightsTemp)

            for i in range(numberOfClusters):
                centroids[i] = [np.sum(weightsTemp[:,i]*X[:,0]),np.sum(weightsTemp[:,i]*X[:,1])]/np.sum(weightsTemp[:,i])

            if (np.linalg.norm(weights - weightsTemp) < epsilon):
                break
            else:
                weights = weightsTemp

        weights = weightsTemp # result
        df = DataFrame(dict(x=X[:,0], y=X[:,1]))
        fig, ax = plt.subplots()
        df.plot(ax=ax, kind='scatter', x='x', y='y', color=weights)
        return weights

