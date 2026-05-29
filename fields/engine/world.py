from dataclasses import dataclass, field

from .time import Time
from .area import Area
from fields.earth import __default_models__ as earth_models

@dataclass
class WorldConfig:
    size_exponent: int = 9
    max_altitude : float = 1000.0 #max altitude in meters
    max_size : float = 200_000.0 #world size in meters
    
    latitude : float = 41. #latitude in degrees, used for temperature gradient and climate
    longitude : float = 29.
    

    sea_level_percentile : float = 0.25 #percentile for sea level
    sea_level : float = None #computed sea level based on height map and percentile
    seed : int = 3563
    debug : bool = False


    init_models : list = field(default_factory=lambda: [Time] + earth_models + [])#, Wind, Humidity, Erosion])

    def __post_init__(self):
        self.size = (2 ** self.size_exponent + 1, 2 ** self.size_exponent + 1)

class World():
    def __init__(self, worldConfig = None):
        if worldConfig is None:
            worldConfig = WorldConfig()
        self.__dict__.update(worldConfig.__dict__)
        
        self.area = Area(self, size = self.size)
        
        self.map_info = {}
        self.models = {}
        self._init_models()

    def _init_models(self):
        for model in self.init_models:
            map_info = model.info.get('map_info', {})
            for map_name, info in map_info.items():
                if map_name in self.map_info:
                    raise ValueError(f"Map name '{map_name}' from model '{model.__name__}' conflicts with an existing map. Please ensure all models have unique map names.")
                self.map_info[map_name] = info

            m = model(self) # initialize the model 
            self.models[m.info['name']] = m
            


    def __call__(self, hours = 1):
        for _ in range(hours):
            for model in self.models.values():
                model()
    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        

