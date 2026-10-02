"""Shared fixtures.

The geometries here deliberately give the wavelength and camera length as
literal values.  A geometry which reads them from file headers cannot be
turned into an Image without a data file, which is exactly the failure mode
the bindings are expected to report clearly.
"""

import numpy as np
import pytest

import crystfel

#: One flat 2048x2048 panel of 100 um pixels, 100 mm downstream, 1 A.
SIMPLE_GEOM = """
wavelength = 1.0 A
clen = 100 mm
adu_per_photon = 1
res = 10000
p0/min_fs = 0
p0/max_fs = 2047
p0/min_ss = 0
p0/max_ss = 2047
p0/corner_x = -1024
p0/corner_y = -1024
p0/fs = x
p0/ss = y
p0/data = /data/data
"""

#: Four panels arranged in a two-level group hierarchy.
GROUPED_GEOM = """
wavelength = 1.0 A
clen = 100 mm
adu_per_photon = 1
res = 10000
data = /data/data

p0/min_fs = 0
p0/max_fs = 511
p0/min_ss = 0
p0/max_ss = 511
p0/corner_x = -520
p0/corner_y = -520
p0/fs = x
p0/ss = y

p1/min_fs = 512
p1/max_fs = 1023
p1/min_ss = 0
p1/max_ss = 511
p1/corner_x = 8
p1/corner_y = -520
p1/fs = x
p1/ss = y

p2/min_fs = 0
p2/max_fs = 511
p2/min_ss = 512
p2/max_ss = 1023
p2/corner_x = -520
p2/corner_y = 8
p2/fs = x
p2/ss = y

p3/min_fs = 512
p3/max_fs = 1023
p3/min_ss = 512
p3/max_ss = 1023
p3/corner_x = 8
p3/corner_y = 8
p3/fs = x
p3/ss = y

group_lower = p0,p1
group_upper = p2,p3
group_all = lower,upper
"""


@pytest.fixture
def dtempl():
    return crystfel.DataTemplate.from_string(SIMPLE_GEOM)


@pytest.fixture
def grouped():
    return crystfel.DataTemplate.from_string(GROUPED_GEOM)


@pytest.fixture
def cell():
    return crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def synthetic_peaks(dtempl, cell, rng, hmax=14, excitation=2.0e7):
    """Peak positions for a randomly oriented crystal of *cell*.

    Reflections close enough to the Ewald sphere are projected onto the
    detector, giving the kind of peak list a real peak finder would produce.
    Returns ``(fs, ss)`` in panel coordinates for panel 0.
    """
    wavelength = dtempl.wavelength
    panel = dtempl.detector()[0]
    pitch = panel.pixel_pitch
    cnx, cny, cnz = panel.corner

    recip = (cell.reciprocal_axes * 1e10) @ random_rotation(rng).T

    h, k, l = np.mgrid[-hmax:hmax + 1, -hmax:hmax + 1,
                       -hmax:hmax + 1].reshape(3, -1)
    nonzero = (h != 0) | (k != 0) | (l != 0)
    h, k, l = h[nonzero], k[nonzero], l[nonzero]

    q = h[:, None] * recip[0] + k[:, None] * recip[1] + l[:, None] * recip[2]
    # Elastic scattering requires |q + s0| = 1/lambda, i.e. qz = -lambda q^2/2
    near_sphere = np.abs(q[:, 2] + wavelength * (q ** 2).sum(1) / 2.0) < excitation
    q = q[near_sphere]

    s = q + np.array([0.0, 0.0, 1.0 / wavelength])
    s /= np.linalg.norm(s, axis=1)[:, None]
    s = s[s[:, 2] > 0.3]

    t = (cnz * pitch) / s[:, 2]
    fs = s[:, 0] * t / pitch - cnx
    ss = s[:, 1] * t / pitch - cny

    height, width = panel.shape
    on_panel = (fs > 0) & (fs < width) & (ss > 0) & (ss < height)
    return fs[on_panel], ss[on_panel]
