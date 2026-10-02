"""Python bindings for libcrystfel.

These wrap the geometry and indexing parts of CrystFEL's shared library.  They
are thin: objects hold a pointer to the corresponding C structure and free it
when they are collected, and the C declarations are read from the installed
headers at build time, so there is no hand-maintained copy of any structure
layout.

Units are not uniform, because CrystFEL's are not:

* :mod:`crystfel.cell` works in angstroms and degrees, like a ``.cell`` file.
* :mod:`crystfel.geometry` and :mod:`crystfel.image` work in metres, inverse
  metres and radians, like a ``.geom`` file -- except for
  :meth:`~crystfel.geometry.DataTemplate.rotate_group`, which takes degrees.

Start here:

>>> import crystfel
>>> dt = crystfel.DataTemplate('detector.geom')
>>> cell = crystfel.UnitCell(63, 63, 42, 90, 90, 120,
...                          lattice_type='hexagonal', unique_axis='c')
>>> indexer = crystfel.Indexer('xgandalf', dt, cell)
>>> img = crystfel.Image(dt)
>>> img.set_peaks(fs, ss, pn, intensity)
>>> if indexer.index(img):
...     print(img.crystals[0].cell)
"""

from crystfel import _log as _log_module

_log_module.install()

from crystfel.errors import CrystFELError, GeometryError, IndexingError
from crystfel.cell import LATTICE_TYPES, UnitCell
from crystfel.geometry import DataTemplate, Detector, GroupInfo, Panel
from crystfel.image import PEAK_DTYPE, Crystal, Image
from crystfel.indexing import Indexer, detect_methods

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("crystfel")
except Exception:          # pragma: no cover - source tree / no metadata
    __version__ = "unknown"

__all__ = [
    "CrystFELError", "GeometryError", "IndexingError",
    "UnitCell", "LATTICE_TYPES",
    "DataTemplate", "Detector", "Panel", "GroupInfo",
    "Image", "Crystal", "PEAK_DTYPE",
    "Indexer", "detect_methods",
    "__version__",
]
