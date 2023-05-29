import matplotlib.pyplot as plt
from TerrainAnalyser.CroppedTerrainImage import CroppedTerrainImage
from typing import List

class TerrainMap:

    def show_terrain_map(self, cropped_terrain_images : List[CroppedTerrainImage], x, y):
        matrix = []
        index = 0
        for i in range(0, y):
            values = []
            for i in range(0, x):
                values.append(cropped_terrain_images[index].get_terrain_category())
                index = index + 1
            matrix.append(values)

        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.Blues)
        plt.show()
        
