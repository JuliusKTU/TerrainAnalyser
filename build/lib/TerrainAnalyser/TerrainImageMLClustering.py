from TerrainAnalyser.TerrainImageAnalyser import TerrainImageAnalyser
from TerrainAnalyser.KMeans import KMeansCustom

class TerrainImageMLClustering(TerrainImageAnalyser):
    
    def __init__(self, x_values, number_of_clusters):
        self.analyser_method = KMeansCustom(x_values, number_of_clusters)
    
    def analyse_terrain_image(self):
        return self.analyser_method.analyse_terrain_image()
    