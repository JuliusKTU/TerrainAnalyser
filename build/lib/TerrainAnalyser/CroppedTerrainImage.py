class CroppedTerrainImage:
    def __init__(self, partition_x, partition_y, terrain_category):
        self.partition_x = partition_x
        self.partition_y = partition_y
        self.terrain_category = terrain_category
        
    def get_terrain_category(self):
        return self.terrain_category