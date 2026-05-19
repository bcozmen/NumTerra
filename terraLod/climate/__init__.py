"""
terraLod.climate — climate simulation sub-package.

Public surface
--------------
    Climate   — full-resolution climate simulator

Internal modules
----------------
    physics   — pure stateless functions (wind, temperature, saturation)
    moisture  — moisture-source building and Numba advection kernel
    climate   — Climate class (orchestration)
"""

from .climate import Climate

__all__ = ['Climate']
