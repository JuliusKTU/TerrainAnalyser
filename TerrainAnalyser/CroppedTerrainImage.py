class CroppedTerrainImage:
    def __init__(self, x_range, y_range, terrain_category):
        self.x_range = x_range
        self.y_range = y_range
        self.terrain_category = terrain_category
        
    def get_terrain_category(self):
        return self.terrain_category