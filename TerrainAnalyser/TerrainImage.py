import os
from pydoc import describe
import cv2
import glob
import re
from sklearn.preprocessing import StandardScaler 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sns
from .TerrainImageMLClustering import TerrainImageMLClustering
from .TerrainCategory import TerrainCategory
from .CroppedTerrainImage import CroppedTerrainImage
from .TerrainMap import TerrainMap

class _TerrainImage:
    
    NUMBER_OF_CATEGORIES = len(TerrainCategory)
    
    def __init__(self):
        self.terrain_coordinates = ""
        self.scale = ""
        self.path = ""
        self.cropped_terrain_images = []
        self.x_range = 0
        self.y_range = 0
        self.terrain_map = TerrainMap()
        
    def provide_image(self, path, terrain_coordinates, scale):
        self.terrain_coordinates = terrain_coordinates
        self.scale = scale
        self.path = glob.glob(path)[0]
        img_id = 0
        files = glob.glob('cropped_images/*')
        for f in files:
            os.remove(f)

        xIndex = 0
        yIndex = 0
        
        if not os.path.exists("cropped_images"):
            os.makedirs("cropped_images")
        
        img = cv2.imread(self.path)
        height, width, channels = img.shape
        print(self.path + " dimensions: " , height , " x " ,width)
        step = 50 # croped image width / height - must be selected under expert evaluation
        for h in range(0, height, step):
            print("Img_id: ", img_id)
            for w in range(0, width, step):
                xIndex = xIndex + 1
                crop_img = img[h:h + step, w:w + step] # cropping image
                img_id += 1
                img_name = "croped_" + str(img_id)
                crop_img_file = "cropped_images/" + img_name + ".jpg"
                cv2.imwrite(crop_img_file, crop_img)
            yIndex = yIndex + 1

        self.x_range = int(xIndex/yIndex)
        self.y_range = int(yIndex)

    def __atoi(self, text):
        return int(text) if text.isdigit() else text
    
    def __natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self.__atoi(c) for c in re.split(r'(\d+)', text) ]
        
    def __extractFeatures(self):
        f = open("dataset.txt", "w")
        imgNr = 1
        mainFiles = glob.glob("cropped_images/*")
        values = []
        mainFiles.sort(key=self.__natural_keys)
        for mainFile in mainFiles:
            head, tail = os.path.split(mainFile)
            f.write(tail)
            img = cv2.imread(mainFile)
            color = ('b', 'g', 'r')
            colorMaxs = [0, 0, 0]
            color_vals = []
            for i, col in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                colorMaxs[0] = hist.max()
                color_vals.append(hist.max())
                fourier = np.fft.fftshift(np.fft.fft2(img)).real.astype(np.float32)
                fourier = fourier.reshape(-1)
                color_vals.append(fourier.mean())
                color_vals.append(fourier.max())
                color_vals.append(fourier.min())
                f.write("\t" + str(fourier.mean()))
                f.write("\t" + str(fourier.max()))
                f.write("\t" + str(fourier.min()))
                
            values.append(color_vals)

            f.write("\n")
        f.close()
        return np.array(values)
    
    def __analyse_terrain_image(self):
        self.cropped_terrain_images = []
        features = self.__extractFeatures()
        clustering_method = TerrainImageMLClustering(features, self.NUMBER_OF_CATEGORIES)
        clustering = clustering_method.analyse_terrain_image()
        for i in range(len(clustering["x"])):
            cropped_terrain_image = CroppedTerrainImage(clustering["x"][i], clustering["y"][i], clustering["label"][i])
            self.cropped_terrain_images.append(cropped_terrain_image)
        
    def get_terrain_analysis_map(self):
        self.__analyse_terrain_image()
        self.terrain_map.show_terrain_map(self.cropped_terrain_images, self.x_range, self.y_range)
        
    def get_terrain_analysis_statistics(self):
        self.__analyse_terrain_image()
        print(self.terrain_map.get_terrain_analysis_statistics(self.cropped_terrain_images, self.x_range, self.y_range))


terrainImage = _TerrainImage()
def provide_image(path, terrain_coordinates, scale):
    terrainImage.provide_image(path, terrain_coordinates, scale)

def get_terrain_analysis_map():
    terrainImage.get_terrain_analysis_map()
    
    
def get_terrain_analysis_statistics():
    terrainImage.get_terrain_analysis_statistics()
