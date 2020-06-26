"""
*
*  o-o          o           o     o
* |     o       |    o      |     |
*  o-o    o-O-o |      o--o O--o -o-
*     | | | | | |    | |  | |  |  |
* o--o  | o o o O---o| o--O o  o  o
*                         |
*                      o--o
*
* Copyright @ Zhou Xiang
"""

from ._version import __version__
from .field import Field, PlaneWave, SphericalWave, Gaussian
from .lens import Lens, CylindricalLens
from .propagation import propagation, near_field_propagation
from .calc import phase, intensity, aberration
from .aperture import circle_aperture
from . import plottools, zernike

sl_version = __version__
print('SimLight ' + sl_version + '\n')
