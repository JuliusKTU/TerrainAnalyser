from sklearn.datasets import (make_blobs, make_circles, make_moons)
import matplotlib.pyplot as plt
from pandas import DataFrame
import random
import numpy as np
from matplotlib.animation import FuncAnimation
from .TerrainImageAnalyser import TerrainImageAnalyser as ta

class KMeansCustom(ta):
    def __init__(self, Xvalues, number_of_clusters):
        self.Xvalues = Xvalues
        self.number_of_clusters = number_of_clusters
    
    def analyse_terrain_image(self):
        X = self.Xvalues
        y = list(0 for i in range (0, len(X)))
        X[:,0]=(X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
        X[:,1]=(X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
        colors = {0:'red', 1:'cyan', 2:'green',3:'cyan', 4:'blue', 5:'yellow', 6:'orange', 7:'brown', 8:'black', 9:'purple', 10:'magenta'}
        number_of_clusters = self.number_of_clusters
        
        print('Initial centroids')
        fig, ax =  plt.subplots()
        centroids = dict()
        for i in range(0,number_of_clusters):
            centroids[i]= 2*np.cos(i*2*np.pi/number_of_clusters),2*np.sin(i*2*np.pi/number_of_clusters)
        clusterLabel = np.zeros((len(y),)) # zeros represent that initally all points are assigned to first cluster (label 0)
        centerX = centroids[0]
        # distances between points and first center
        dist = np.sqrt((X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2) # each point distance to the first cluster
        for k in range(1, number_of_clusters):
            centerX = centroids[k]
            distTemp = np.sqrt((X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2) # distance to k cluster
            # assigning points to k cluster (if distance is shorter than current)
            mask = distTemp < dist
            dist[mask] = distTemp[mask]
            clusterLabel[mask] = k

        # creating dataframe
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=clusterLabel))
        grouped = df.groupby('label')
        for key, group in grouped:
            xc = group['x'].mean()
            yc = group['y'].mean()
            centroids[key] = xc, yc
          
        numberOfIterations = 10 # max number of iteration
        for i in range(0,numberOfIterations):

            df = DataFrame(dict(x=X[:,0], y=X[:,1], label=clusterLabel))
            grouped = df.groupby('label')

            # getting new centers
            for key, group in grouped:
                xc = group['x'].mean()
                yc = group['y'].mean()
                centroids[key] = xc, yc
                
            # reasinging data to new clusters
            clusterLabelTemp = np.zeros_like(clusterLabel)
            centerX = centroids[0]
            dist = np.sqrt((X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2)
            for k in range(1, number_of_clusters):
                centerX = centroids[k]
                distTemp = np.sqrt((X[:,0]-centerX[0])**2 + (X[:,1]-centerX[1])**2)
                mask = distTemp < dist
                dist[mask] = distTemp[mask]
                clusterLabelTemp[mask] = k

            # if no changes - stop computation
            if np.count_nonzero(clusterLabel != clusterLabelTemp) == 0:
                break
            clusterLabel = clusterLabelTemp
        
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=clusterLabel))
        return df
