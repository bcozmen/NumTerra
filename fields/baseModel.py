from abc import ABC, abstractmethod

class BaseModel(ABC):
    info = None
    def __init__(self, world, config = None):
        self.world = world
        self.config = config
        if self.info is None:
            raise ValueError("Model must have an 'info' attribute with metadata about the model and its maps.")
        


    def __call__(self, area = None):
        if area is None:
            self.step()
        else:
            self.generate(area)

    ## ========== Simulation & Generation ==========
    def init(self):
        pass

    def step(self, **kwargs):
        pass

    def generate(self, area):
        pass


    ## ========== Maps Management ==========
    def set_maps(self, maps_dict):
        for key, value in maps_dict.items():
            self.world.area[key] = value
    
    def get_maps(self):
        pass

    ## ========== Utility Methods ==========
    def _step(self):
        pass

    def _init(self):
        pass

    

