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
from . import plottools
from . import calc

sl_version = __version__
print('SimLight ' + sl_version + '\n')
