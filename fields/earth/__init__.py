from .terrain.terrain import Terrain
from .climate.prognostic import PrognosticClimate
from .climate.diagnostic import DiagnosticClimate

__default_models__ = [Terrain, PrognosticClimate, DiagnosticClimate]
__all__ = ["Terrain", "PrognosticClimate", "DiagnosticClimate"]