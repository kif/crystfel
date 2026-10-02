# cython: language_level=3
# cython: boundscheck=False, wraparound=False
"""Diffraction patterns and the crystals found in them.

An :class:`Image` built from a :class:`~crystfel.geometry.DataTemplate` has
zeroed data arrays and a detector, which is all the indexing engines need: you
supply the peak list yourself, from whatever peak finder you like.

Peak coordinates follow the CrystFEL convention, which is easy to get wrong:
``fs`` and ``ss`` are **distances in pixels from the corner of the panel**, not
pixel indices.  A peak at the centre of the first pixel of a panel is at
0.5, 0.5.  Getting this wrong by half a pixel is a common reason for indexing
to quietly stop working.
"""

import numpy as np

from crystfel import _log
from crystfel.errors import CrystFELError

from crystfel.cell cimport UnitCell, wrap_cell
from crystfel.geometry cimport DataTemplate, wrap_detector

from crystfel._libcrystfel cimport (Crystal as c_Crystal, ImageFeatureList,
                                    crystal_get_cell, crystal_get_det_shift,
                                    crystal_get_mosaicity,
                                    crystal_get_notes,
                                    crystal_get_profile_radius,
                                    crystal_get_resolution_limit,
                                    crystal_get_user_flag, image as c_image,
                                    image_add_feature, image_create_for_simulation,
                                    image_feature_count,
                                    image_feature_list_free,
                                    image_feature_list_new, image_free,
                                    image_get_feature, imagefeature,
                                    indexer_str)

#: dtype of the array returned by :attr:`Image.peaks`.
PEAK_DTYPE = np.dtype([("fs", "f8"), ("ss", "f8"),
                       ("pn", "i4"), ("intensity", "f8")])


cdef class Image:
    """One diffraction pattern.

    >>> img = Image(dtempl)
    >>> img.set_peaks(fs, ss, pn, intensity)
    >>> n = indexer.index(img)
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._dtempl = None

    def __init__(self, DataTemplate dtempl not None):
        self._c = image_create_for_simulation(dtempl._c)
        if self._c is NULL:
            raise CrystFELError(_log.explain(
                "Could not create image from geometry.  The geometry must "
                "give the wavelength and camera length as literal values, "
                "not as references to file headers"))
        self._dtempl = dtempl

    def __dealloc__(self):
        if self._c is not NULL:
            image_free(self._c)
        self._c = NULL

    # ----------------------------------------------------------- parameters

    @property
    def wavelength(self):
        """Wavelength of the incident radiation, in metres."""
        return self._c.wavelength

    @wavelength.setter
    def wavelength(self, double value):
        self._c.wavelength = value

    @property
    def bandwidth(self):
        """Fractional FWHM bandwidth."""
        return self._c.bw

    @bandwidth.setter
    def bandwidth(self, double value):
        self._c.bw = value

    @property
    def divergence(self):
        """Full convergence angle of the beam, in radians."""
        return self._c.div

    @divergence.setter
    def divergence(self, double value):
        self._c.div = value

    @property
    def serial(self):
        return self._c.serial

    @serial.setter
    def serial(self, int value):
        self._c.serial = value

    @property
    def hit(self):
        return bool(self._c.hit)

    @hit.setter
    def hit(self, value):
        self._c.hit = 1 if value else 0

    @property
    def peak_resolution(self):
        """Resolution estimate from the peaks (1/d in inverse metres), or
        ``None`` if it has not been calculated."""
        if self._c.peak_resolution < 0.0:
            return None
        return self._c.peak_resolution

    @property
    def detector(self):
        """The :class:`~crystfel.geometry.Detector` for this image."""
        return wrap_detector(self._c.det, False, self)

    # ---------------------------------------------------------------- peaks

    cdef ImageFeatureList *_features(self) except NULL:
        if self._c.features is NULL:
            self._c.features = image_feature_list_new()
            if self._c.features is NULL:
                raise MemoryError("Could not allocate peak list")
            # image_free() will free it, as owns_peaklist is set by image_new()
        return self._c.features

    @property
    def n_peaks(self):
        if self._c.features is NULL:
            return 0
        return image_feature_count(self._c.features)

    def add_peak(self, double fs, double ss, int pn, double intensity=0.0,
                 name=None):
        """Add a single peak.  See the module docstring about ``fs``/``ss``."""
        cdef bytes raw
        cdef ImageFeatureList *flist = self._features()
        if name is None:
            image_add_feature(flist, fs, ss, pn, intensity, NULL)
        else:
            raw = str(name).encode("utf-8")
            image_add_feature(flist, fs, ss, pn, intensity, raw)

    def set_peaks(self, fs, ss, pn, intensity=None):
        """Replace the peak list with the contents of the given arrays.

        All arguments are broadcast against each other, so a single panel
        number may be given for the whole list.
        """
        cdef Py_ssize_t i, n
        cdef const double[::1] vfs
        cdef const double[::1] vss
        cdef const int[::1] vpn
        cdef const double[::1] vint
        cdef ImageFeatureList *flist

        if intensity is None:
            intensity = 0.0
        afs, ass, apn, aint = np.broadcast_arrays(
            np.asarray(fs, dtype=np.float64),
            np.asarray(ss, dtype=np.float64),
            np.asarray(pn, dtype=np.int32),
            np.asarray(intensity, dtype=np.float64))

        vfs = np.ascontiguousarray(afs).reshape(-1)
        vss = np.ascontiguousarray(ass).reshape(-1)
        vpn = np.ascontiguousarray(apn).reshape(-1)
        vint = np.ascontiguousarray(aint).reshape(-1)
        n = vfs.shape[0]

        self.clear_peaks()
        flist = self._features()
        with nogil:
            for i in range(n):
                image_add_feature(flist, vfs[i], vss[i], vpn[i], vint[i],
                                  NULL)

    def clear_peaks(self):
        """Discard the peak list."""
        if self._c.features is not NULL:
            image_feature_list_free(self._c.features)
            self._c.features = NULL

    @property
    def peaks(self):
        """The peak list, as a structured array with fields
        ``fs``, ``ss``, ``pn`` and ``intensity``."""
        cdef Py_ssize_t i, n = self.n_peaks
        cdef imagefeature *f
        cdef ImageFeatureList *flist = self._c.features
        cdef double[::1] vfs
        cdef double[::1] vss
        cdef int[::1] vpn
        cdef double[::1] vint

        out = np.zeros(n, dtype=PEAK_DTYPE)
        if n == 0:
            return out

        afs = np.zeros(n, dtype=np.float64)
        ass = np.zeros(n, dtype=np.float64)
        apn = np.zeros(n, dtype=np.int32)
        aint = np.zeros(n, dtype=np.float64)
        vfs = afs
        vss = ass
        vpn = apn
        vint = aint

        with nogil:
            for i in range(n):
                f = image_get_feature(flist, i)
                if f is NULL:
                    continue
                vfs[i] = f.fs
                vss[i] = f.ss
                vpn[i] = f.pn
                vint[i] = f.intensity

        out["fs"] = afs
        out["ss"] = ass
        out["pn"] = apn
        out["intensity"] = aint
        return out

    # ------------------------------------------------------------- crystals

    @property
    def n_crystals(self):
        return self._c.n_crystals

    @property
    def crystals(self):
        """The crystals found by indexing, as a tuple of :class:`Crystal`."""
        return tuple(_wrap_crystal(self._c.crystals[i].cr, self)
                     for i in range(self._c.n_crystals))

    @property
    def indexed_by(self):
        """Name of the indexing method which indexed this pattern."""
        return indexer_str(self._c.indexed_by).decode("utf-8")

    @property
    def n_indexing_tries(self):
        return self._c.n_indexing_tries

    def __repr__(self):
        return ("<Image serial %i, %i peaks, %i crystals>"
                % (self._c.serial, self.n_peaks, self._c.n_crystals))


cdef Crystal _wrap_crystal(c_Crystal *cr, object parent):
    cdef Crystal out
    if cr is NULL:
        return None
    out = Crystal.__new__(Crystal)
    out._c = cr
    out._parent = parent
    return out


cdef class Crystal:
    """One indexing solution.

    This is a view onto a crystal owned by an :class:`Image`, and keeps that
    image alive for as long as it exists.
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._parent = None

    def __init__(self):
        raise TypeError("Crystal objects come from Image.crystals")

    @property
    def cell(self):
        """The indexed unit cell."""
        return wrap_cell(crystal_get_cell(self._c), False, self)

    @property
    def det_shift(self):
        """``(dx, dy)`` beam centre shift refined for this crystal, in metres."""
        cdef double dx, dy
        crystal_get_det_shift(self._c, &dx, &dy)
        return (dx, dy)

    @property
    def profile_radius(self):
        """Reciprocal-space profile radius, in inverse metres."""
        return crystal_get_profile_radius(self._c)

    @property
    def resolution_limit(self):
        """Resolution limit (1/d) in inverse metres.

        Only meaningful once the reflections have been integrated; after
        indexing alone it is whatever CrystFEL left it at.
        """
        return crystal_get_resolution_limit(self._c)

    @property
    def mosaicity(self):
        return crystal_get_mosaicity(self._c)

    @property
    def user_flag(self):
        return crystal_get_user_flag(self._c)

    @property
    def notes(self):
        cdef const char *n = crystal_get_notes(self._c)
        return None if n is NULL else n.decode("utf-8")

    def __repr__(self):
        return "<Crystal %r>" % (self.cell,)
