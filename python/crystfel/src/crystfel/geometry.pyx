# cython: language_level=3
# cython: boundscheck=False, wraparound=False
"""Detector geometry: reading ``.geom`` files, moving panels around, and
converting between coordinate systems.

Units
-----
Everything here is SI, because that is what ``.geom`` files and libcrystfel
use internally: distances in **metres**, wavelengths in **metres**, scattering
vectors in **inverse metres**.  Angles passed to :meth:`DataTemplate.rotate_group`
are in **degrees** for convenience (libcrystfel wants radians, and the
conversion happens here, exactly as ``adjust_detector`` does it).

Note that :mod:`crystfel.cell` uses angstroms and degrees instead, matching
the convention of ``.cell`` files.

Two different objects describe the detector:

:class:`DataTemplate`
    The parsed ``.geom`` file.  It describes *how to read data*: which panel
    lives where in the file, where the headers are, the group hierarchy, and
    so on.  This is the thing you transform when aligning a detector.

:class:`Detector`
    The concrete geometry (``struct detgeom``) for one frame, derived from a
    :class:`DataTemplate`.  This is what you compute with.
"""

from libc.math cimport M_PI
from libc.stdio cimport FILE, fclose, fflush
from libc.stdlib cimport free

import collections

import numpy as np

from crystfel import _log
from crystfel.errors import GeometryError

from crystfel._libcrystfel cimport (DataTemplate as c_DataTemplate, cffree,
                                    data_template_add_to_scanv_beamxy,
                                    data_template_file_to_panel_coords,
                                    data_template_free,
                                    data_template_get_clen_if_possible,
                                    data_template_get_detgeom_if_possible,
                                    data_template_get_slab_extents,
                                    data_template_get_wavelength_if_possible,
                                    data_template_group_info,
                                    data_template_new_from_file,
                                    data_template_new_from_string,
                                    data_template_panel_name_to_number,
                                    data_template_panel_number_to_name,
                                    data_template_panel_to_file_coords,
                                    data_template_print_total_movements,
                                    data_template_reset_total_movements,
                                    data_template_rotate_group,
                                    data_template_show_hierarchy,
                                    data_template_slabby_file_to_panel_coords,
                                    data_template_translate_group_m,
                                    data_template_translate_group_px,
                                    data_template_write_to_fh,
                                    data_template_write_to_file,
                                    detgeom as c_detgeom,
                                    detgeom_find_panel, detgeom_free,
                                    detgeom_group_center,
                                    detgeom_max_resolution,
                                    detgeom_mean_camera_length,
                                    detgeom_panel as c_detgeom_panel,
                                    detgeom_show_hierarchy,
                                    detgeom_transform_coords,
                                    detgeom_translate_detector_m,
                                    dg_group_info)

cdef extern from "stdio.h" nogil:
    FILE *open_memstream(char **ptr, size_t *sizeloc)

cdef double _RAD = M_PI / 180.0


#: One entry of the panel group hierarchy of a :class:`DataTemplate`.
GroupInfo = collections.namedtuple("GroupInfo",
                                   "name serial hierarchy_level leaf")


cdef class DataTemplate:
    """A parsed CrystFEL geometry file.

    >>> dt = DataTemplate('detector.geom')
    >>> dt.translate_group('q0', 0.0, 0.0, 1e-4)      # 100 um along z
    >>> dt.write('detector-moved.geom')
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL

    def __init__(self, filename):
        cdef bytes raw = str(filename).encode("utf-8")
        self._c = data_template_new_from_file(raw)
        if self._c is NULL:
            raise GeometryError(_log.explain("Could not read geometry from %s"
                                             % filename))

    def __dealloc__(self):
        if self._c is not NULL:
            data_template_free(self._c)
        self._c = NULL

    @classmethod
    def from_string(cls, text):
        """Parse a geometry file which is already in memory."""
        cdef bytes raw = text.encode("utf-8") if isinstance(text, str) else text
        cdef DataTemplate out = DataTemplate.__new__(DataTemplate)
        out._c = data_template_new_from_string(raw)
        if out._c is NULL:
            raise GeometryError(_log.explain("Could not parse geometry"))
        return out

    # ------------------------------------------------------------- writing

    def write(self, filename):
        """Write the (possibly transformed) geometry to a file."""
        cdef bytes raw = str(filename).encode("utf-8")
        if data_template_write_to_file(self._c, raw):
            raise GeometryError(_log.explain("Could not write geometry to %s"
                                             % filename))

    def to_string(self):
        """Return the geometry as text, exactly as :meth:`write` would save it."""
        cdef char *buf = NULL
        cdef size_t size = 0
        cdef FILE *fh = open_memstream(&buf, &size)
        cdef int err
        if fh is NULL:
            raise MemoryError("Could not open memory stream")
        try:
            err = data_template_write_to_fh(self._c, fh)
            fflush(fh)
            if err:
                raise GeometryError(_log.explain("Could not serialise geometry"))
            return buf[:size].decode("utf-8")
        finally:
            fclose(fh)
            free(buf)

    # ------------------------------------------------------------- queries

    @property
    def wavelength(self):
        """Wavelength in metres, or ``None`` if the geometry does not fix it
        (because it is read from a file header)."""
        cdef double wl = data_template_get_wavelength_if_possible(self._c)
        return None if wl != wl else wl        # NaN means "not possible"

    @property
    def camera_length(self):
        """Camera length in metres, or ``None`` if it comes from a header."""
        cdef double clen = data_template_get_clen_if_possible(self._c)
        return None if clen != clen else clen

    @property
    def slab_extents(self):
        """``(width, height)`` of the data block, or ``None`` if the panels do
        not all come from one 2D slab."""
        cdef int w, h
        if data_template_get_slab_extents(self._c, &w, &h):
            return None
        return (w, h)

    def panel_name(self, int pn):
        """The name of panel number *pn*."""
        cdef const char *name = data_template_panel_number_to_name(self._c, pn)
        if name is NULL:
            raise IndexError("No panel number %i" % pn)
        return name.decode("utf-8")

    def panel_number(self, name):
        """The number of the panel called *name*."""
        cdef bytes raw = str(name).encode("utf-8")
        cdef int pn
        if data_template_panel_name_to_number(self._c, raw, &pn):
            raise KeyError("No panel called %r" % name)
        return pn

    @property
    def groups(self):
        """The panel group hierarchy, as a list of :class:`GroupInfo`.

        The list is in depth-first order, so ``hierarchy_level`` can be used
        directly for indenting.  These names are what you pass to
        :meth:`translate_group` and :meth:`rotate_group`.
        """
        cdef int n = 0
        cdef dg_group_info *info = data_template_group_info(self._c, &n)
        if info is NULL:
            raise GeometryError(_log.explain("Could not read group hierarchy"))
        try:
            return [GroupInfo(info[i].name.decode("utf-8"),
                              info[i].serial,
                              info[i].hierarchy_level,
                              bool(info[i].leaf))
                    for i in range(n)]
        finally:
            cffree(info)

    def show_hierarchy(self):
        """Print the group hierarchy, as ``adjust_detector --hierarchy`` does."""
        data_template_show_hierarchy(self._c)

    def detector(self, no_clen_ok=False):
        """Build the concrete :class:`Detector` for this geometry.

        Returns ``None`` if the geometry does not pin down the detector
        position without a data file (for example when ``clen`` is read from a
        header).  Pass ``no_clen_ok=True`` to accept a detector whose camera
        length is unknown, which is enough for anything which only needs the
        panel layout.
        """
        cdef c_detgeom *dg
        dg = data_template_get_detgeom_if_possible(self._c, 1 if no_clen_ok
                                                   else 0)
        if dg is NULL:
            return None
        return wrap_detector(dg, True, None)

    # ----------------------------------------------------- transformations

    def translate_group(self, name, double x, double y, double z,
                        units='m'):
        """Move a panel group.

        *units* is ``'m'`` for metres or ``'px'`` for pixels.  Pixel moves are
        only meaningful when every panel in the group has the same pixel size.
        """
        cdef bytes raw = str(name).encode("utf-8")
        cdef int err
        if units == 'm':
            err = data_template_translate_group_m(self._c, raw, x, y, z)
        elif units == 'px':
            err = data_template_translate_group_px(self._c, raw, x, y, z)
        else:
            raise ValueError("units must be 'm' or 'px', not %r" % units)
        if err:
            raise GeometryError(_log.explain("Could not translate group %r"
                                             % name))

    def rotate_group(self, name, double angle, axis):
        """Rotate a panel group by *angle* **degrees** about *axis*.

        *axis* is ``'x'``, ``'y'`` or ``'z'``.  The rotation centre is the
        centre of the group, as reported by :attr:`Detector.group_center`.
        """
        cdef bytes raw = str(name).encode("utf-8")
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y' or 'z', not %r" % axis)
        if data_template_rotate_group(self._c, raw, angle * _RAD,
                                      <char> ord(axis)):
            raise GeometryError(_log.explain("Could not rotate group %r"
                                             % name))

    def add_to_scanv_beamxy(self, updates):
        """Apply a four-element correction to the scan vectors and beam centre.

        This is the update produced by detector alignment; see the
        ``align_detector`` documentation for the meaning of the four values.
        """
        cdef double vals[4]
        if len(updates) != 4:
            raise ValueError("Need exactly four values")
        for i in range(4):
            vals[i] = float(updates[i])
        if data_template_add_to_scanv_beamxy(self._c, vals):
            raise GeometryError(_log.explain("Could not update scan vectors"))

    def reset_total_movements(self):
        """Forget the accumulated panel movements."""
        data_template_reset_total_movements(self._c)

    def print_total_movements(self):
        """Print how far each panel has moved since the last reset."""
        data_template_print_total_movements(self._c)

    # ------------------------------------------------- coordinate transforms

    def file_to_panel_coords(self, fs, ss, int pn):
        """Convert file ("slab") coordinates to coordinates within panel *pn*.

        *fs* and *ss* may be scalars or arrays of any shape.  Note the
        CrystFEL convention: these are distances in pixels from the corner of
        the panel, **not** pixel indices, so the centre of the first pixel is
        at 0.5, 0.5.
        """
        afs, ass, scalar, mfs, mss, n = _prepare_pair(fs, ss)
        cdef float[::1] vfs = mfs
        cdef float[::1] vss = mss
        cdef Py_ssize_t i
        cdef Py_ssize_t count = n
        cdef int bad = 0
        with nogil:
            for i in range(count):
                bad |= data_template_file_to_panel_coords(self._c, &vfs[i],
                                                          &vss[i], pn)
        if bad:
            raise GeometryError("Could not convert coordinates for panel %i"
                                % pn)
        if scalar:
            return float(afs), float(ass)
        return afs, ass

    def panel_to_file_coords(self, int pn, fs, ss):
        """The inverse of :meth:`file_to_panel_coords`."""
        afs, ass, scalar, mfs, mss, n = _prepare_pair(fs, ss)
        cdef float[::1] vfs = mfs
        cdef float[::1] vss = mss
        cdef Py_ssize_t i
        cdef Py_ssize_t count = n
        cdef int bad = 0
        with nogil:
            for i in range(count):
                bad |= data_template_panel_to_file_coords(self._c, pn,
                                                          &vfs[i], &vss[i])
        if bad:
            raise GeometryError("Could not convert coordinates for panel %i"
                                % pn)
        if scalar:
            return float(afs), float(ass)
        return afs, ass

    def slabby_file_to_panel_coords(self, fs, ss):
        """Convert file coordinates to panel coordinates, working out which
        panel each position falls in.

        Returns ``(fs, ss, pn)``.  Only works for geometries where all panels
        come from one 2D slab.
        """
        afs, ass, scalar, mfs, mss, n = _prepare_pair(fs, ss)
        apn = np.empty(n, dtype=np.int32)
        cdef float[::1] vfs = mfs
        cdef float[::1] vss = mss
        cdef int[::1] vpn = apn
        cdef Py_ssize_t i
        cdef Py_ssize_t count = n
        cdef int bad = 0
        with nogil:
            for i in range(count):
                bad |= data_template_slabby_file_to_panel_coords(self._c,
                                                                 &vfs[i],
                                                                 &vss[i],
                                                                 &vpn[i])
        if bad:
            raise GeometryError("Some coordinates do not fall on any panel")
        if scalar:
            return float(afs), float(ass), int(apn[0])
        return afs, ass, apn.reshape(afs.shape)

    def __repr__(self):
        wl = self.wavelength
        return ("<DataTemplate %s, wavelength %s>"
                % (self.slab_extents, "unknown" if wl is None
                   else "%.4f A" % (wl * 1e10)))


cdef _prepare_pair(fs, ss):
    """Make writable, contiguous float32 copies of a coordinate pair."""
    scalar = np.isscalar(fs) and np.isscalar(ss)
    afs = np.array(fs, dtype=np.float32, copy=True)
    ass = np.array(ss, dtype=np.float32, copy=True)
    if afs.shape != ass.shape:
        raise ValueError("fs and ss must have the same shape")
    mfs = afs.reshape(-1)
    mss = ass.reshape(-1)
    return afs, ass, scalar, mfs, mss, mfs.shape[0]


cdef Detector wrap_detector(c_detgeom *dg, bint owned, object parent):
    cdef Detector out
    if dg is NULL:
        return None
    out = Detector.__new__(Detector)
    out._c = dg
    out._owned = owned
    out._parent = parent
    return out


cdef class Detector:
    """The concrete detector geometry for one frame (``struct detgeom``).

    Obtain one from :meth:`DataTemplate.detector` or :attr:`crystfel.Image.detector`.
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._owned = False
        self._parent = None

    def __init__(self):
        raise TypeError("Detector objects come from DataTemplate.detector() "
                        "or Image.detector")

    def __dealloc__(self):
        if self._owned and self._c is not NULL:
            detgeom_free(self._c)
        self._c = NULL

    def __len__(self):
        return self._c.n_panels

    def __getitem__(self, key):
        cdef c_detgeom_panel *p
        cdef bytes raw
        if isinstance(key, str):
            raw = key.encode("utf-8")
            p = detgeom_find_panel(self._c, raw)
            if p is NULL:
                raise KeyError("No panel called %r" % key)
            return _wrap_panel(p, self)
        idx = int(key)
        if idx < 0:
            idx += self._c.n_panels
        if idx < 0 or idx >= self._c.n_panels:
            raise IndexError("No panel number %r" % key)
        return _wrap_panel(&self._c.panels[idx], self)

    @property
    def panels(self):
        return tuple(self[i] for i in range(self._c.n_panels))

    @property
    def mean_camera_length(self):
        """Mean camera length over all panels, in metres."""
        return detgeom_mean_camera_length(self._c)

    def max_resolution(self, double wavelength):
        """Highest 1/d reachable on this detector, in inverse metres."""
        return detgeom_max_resolution(self._c, wavelength)

    def translate(self, double x, double y, double z):
        """Move the whole detector by (x, y, z) metres.

        This changes only this :class:`Detector`.  To change the geometry
        itself, use :meth:`DataTemplate.translate_group`.
        """
        detgeom_translate_detector_m(self._c, x, y, z)

    @property
    def group_center(self):
        """Centre of the top-level group, in metres."""
        cdef double x, y, z
        if self._c.top_group is NULL:
            return None
        if detgeom_group_center(self._c.top_group, &x, &y, &z):
            raise GeometryError("Could not find group centre")
        return (x, y, z)

    def show_hierarchy(self):
        detgeom_show_hierarchy(self._c)

    def transform_coords(self, panel, fs, ss, double wavelength,
                         double dx=0.0, double dy=0.0):
        """Map detector positions to scattering vectors.

        *panel* is a panel number or name.  *fs* and *ss* are positions on
        that panel, in pixels from its corner.  *dx* and *dy* shift the beam
        centre, in pixels (this is where a crystal's detector shift goes).

        Returns an array of shape ``(..., 3)`` holding the scattering vector
        in **inverse metres**, in the lab frame, with the Ewald sphere
        correction already applied (so the direct beam maps to the origin).
        """
        cdef Panel pan = self[panel] if not isinstance(panel, Panel) else panel
        cdef c_detgeom_panel *p = pan._c
        # Note: np.ascontiguousarray promotes 0-d arrays to 1-d, so the
        # output shape has to be taken from asarray(), before that happens.
        afs = np.asarray(fs, dtype=np.float64)
        ass = np.asarray(ss, dtype=np.float64)
        if afs.shape != ass.shape:
            raise ValueError("fs and ss must have the same shape")
        shape = afs.shape
        flat_fs = np.ascontiguousarray(afs).reshape(-1)
        flat_ss = np.ascontiguousarray(ass).reshape(-1)
        out = np.empty((flat_fs.shape[0], 3), dtype=np.float64)
        cdef const double[::1] vfs = flat_fs
        cdef const double[::1] vss = flat_ss
        cdef double[:, ::1] vout = out
        cdef Py_ssize_t i
        cdef Py_ssize_t count = vfs.shape[0]
        with nogil:
            for i in range(count):
                detgeom_transform_coords(p, vfs[i], vss[i], wavelength,
                                         dx, dy, &vout[i, 0])
        return out.reshape(shape + (3,))

    def __repr__(self):
        return "<Detector with %i panels>" % self._c.n_panels


cdef Panel _wrap_panel(c_detgeom_panel *p, Detector det):
    cdef Panel out = Panel.__new__(Panel)
    out._c = p
    out._det = det
    return out


cdef class Panel:
    """One panel of a :class:`Detector`.

    This is a view onto memory owned by the detector; it keeps the detector
    alive for as long as it exists.
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._det = None

    def __init__(self):
        raise TypeError("Panel objects come from a Detector")

    @property
    def name(self):
        return self._c.name.decode("utf-8")

    @property
    def shape(self):
        """``(height, width)`` in pixels, in numpy order."""
        return (self._c.h, self._c.w)

    @property
    def pixel_pitch(self):
        """Pixel size in metres."""
        return self._c.pixel_pitch

    @property
    def adu_per_photon(self):
        return self._c.adu_per_photon

    @property
    def max_adu(self):
        return self._c.max_adu

    @property
    def corner(self):
        """Corner position ``(cnx, cny, cnz)``, in units of this panel's own
        pixel size, measured from the interaction point.  This is the raw
        CrystFEL representation; see :attr:`corner_m` for metres."""
        return (self._c.cnx, self._c.cny, self._c.cnz)

    @property
    def corner_m(self):
        """Corner position in metres."""
        return (self._c.cnx * self._c.pixel_pitch,
                self._c.cny * self._c.pixel_pitch,
                self._c.cnz * self._c.pixel_pitch)

    @property
    def fs_vector(self):
        """Lab-frame direction of one step along the fast-scan axis."""
        return (self._c.fsx, self._c.fsy, self._c.fsz)

    @property
    def ss_vector(self):
        """Lab-frame direction of one step along the slow-scan axis."""
        return (self._c.ssx, self._c.ssy, self._c.ssz)

    @property
    def group(self):
        """Name of the leaf group containing this panel, or ``None``."""
        if self._c.group is NULL or self._c.group.name is NULL:
            return None
        return self._c.group.name.decode("utf-8")

    def __repr__(self):
        return "<Panel %s %ix%i>" % (self.name, self._c.w, self._c.h)
