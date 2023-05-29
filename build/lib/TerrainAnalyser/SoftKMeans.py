
from TerrainAnalyser.TerrainImage import TerrainImage
import numpy as np
from TerrainAnalyser.KMeans import KMeansCustom

terrain = TerrainImage("1", "2")
terrain.provide_image("TerrainAnalyser/Vilnius_Lithuania.tif")
terrain.get_terrain_analysis_map()


# data = features
# kmeans = KMeansCustom(data, 3)
# groups = kmeans.Execute()
# groups = groups["label"]
# drawMatrix(groups, Xrange, Yrange)

