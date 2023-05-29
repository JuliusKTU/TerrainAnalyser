from .TerrainImageAnalyser import TerrainImageAnalyser

class TerrainImageManualClustering(TerrainImageAnalyser):
    
    def __init__(self, x_values, number_of_clusters):
        self.x_values = x_values
        self.number_of_clusters = number_of_clusters
