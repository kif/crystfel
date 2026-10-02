# cython: language_level=3
"""Unit cells.

Units
-----
libcrystfel stores cell lengths in **metres** and angles in **radians**.
This module converts at the boundary: everything in the Python API is in
**angstroms** and **degrees**, which is what you would type into a ``.cell``
file.  Reciprocal-space quantities are in inverse angstroms.
"""

from libc.math cimport M_PI

import numpy as np

from crystfel import _log
from crystfel.errors import CrystFELError

from crystfel._libcrystfel cimport (IntegerMatrix, LatticeType, RationalMatrix,
                                    UnitCell as c_UnitCell, bravais_lattice,
                                    cell_free, cell_get_cartesian,
                                    cell_get_centering, cell_get_lattice_type,
                                    cell_get_parameters, cell_get_reciprocal,
                                    cell_get_unique_axis, cell_get_volume,
                                    cell_has_parameters,
                                    cell_new_from_parameters,
                                    cell_new_from_cell, cell_set_centering,
                                    cell_set_lattice_type,
                                    cell_set_unique_axis,
                                    cell_transform_intmat, cell_is_sensible,
                                    compare_cell_parameters,
                                    compare_reindexed_cell_parameters,
                                    impose_bravais, intmat_create_3x3,
                                    intmat_free, intmat_get,
                                    lattice_from_str, load_cell_from_file,
                                    resolution, right_handed, rtnl_as_double,
                                    rtnl_mtx_free, rtnl_mtx_get, str_lattice,
                                    uncenter_cell, validate_cell,
                                    write_cell_to_file)

cdef double _RAD = M_PI / 180.0

#: The lattice types known to CrystFEL, in CrystFEL's own spelling.
LATTICE_TYPES = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
                 "rhombohedral", "hexagonal", "cubic")


cdef LatticeType _lattice_from_str(name) except *:
    cdef bytes raw = str(name).encode("utf-8")
    if str(name) not in LATTICE_TYPES:
        raise ValueError("Unknown lattice type %r (expected one of %s)"
                         % (name, ", ".join(LATTICE_TYPES)))
    return lattice_from_str(raw)


cdef char _one_char(value, what) except *:
    text = str(value)
    if len(text) != 1:
        raise ValueError("%s must be a single character, got %r"
                         % (what, value))
    return <char> ord(text)


cdef void _fill_tolerances(tolerances, double *tols) except *:
    """Convert a user tolerance 6-tuple (fractions, then degrees) to the
    fractions-and-radians form libcrystfel expects."""
    if len(tolerances) != 6:
        raise ValueError("Need exactly six tolerances")
    for i in range(3):
        tols[i] = float(tolerances[i])
    for i in range(3, 6):
        tols[i] = float(tolerances[i]) * _RAD


cdef UnitCell wrap_cell(c_UnitCell *c, bint owned, object parent):
    """Wrap an existing C unit cell.

    If *owned*, the returned object frees the cell when it is collected.
    Otherwise the cell belongs to *parent* (a Crystal, say), and a reference to
    *parent* is kept so that it cannot be collected first.
    """
    cdef UnitCell out
    if c is NULL:
        return None
    out = UnitCell.__new__(UnitCell)
    out._c = c
    out._owned = owned
    out._parent = parent
    return out


cdef class UnitCell:
    """A crystallographic unit cell.

    >>> cell = UnitCell(63.0, 63.0, 42.0, 90, 90, 120,
    ...                 centering='P', lattice_type='hexagonal',
    ...                 unique_axis='c')
    """

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._owned = False
        self._parent = None

    def __init__(self, a, b, c, alpha, beta, gamma,
                 centering='P', lattice_type='triclinic', unique_axis='*'):
        self._c = cell_new_from_parameters(a * 1e-10, b * 1e-10, c * 1e-10,
                                           alpha * _RAD, beta * _RAD,
                                           gamma * _RAD)
        if self._c is NULL:
            raise CrystFELError(_log.explain("Could not create unit cell"))
        self._owned = True
        cell_set_centering(self._c, _one_char(centering, "centering"))
        cell_set_lattice_type(self._c, _lattice_from_str(lattice_type))
        cell_set_unique_axis(self._c, _one_char(unique_axis, "unique_axis"))

    def __dealloc__(self):
        if self._owned and self._c is not NULL:
            cell_free(self._c)
        self._c = NULL

    # ---------------------------------------------------------- constructors

    @classmethod
    def from_file(cls, filename):
        """Read a cell from a ``.cell`` or PDB file."""
        cdef bytes raw = str(filename).encode("utf-8")
        cdef c_UnitCell *c = load_cell_from_file(raw)
        if c is NULL:
            raise CrystFELError(_log.explain("Could not read cell from %s"
                                             % filename))
        return wrap_cell(c, True, None)

    def write(self, filename):
        """Write the cell to a ``.cell`` file."""
        cdef bytes raw = str(filename).encode("utf-8")
        if write_cell_to_file(self._c, raw):
            raise CrystFELError(_log.explain("Could not write cell to %s"
                                             % filename))

    # ------------------------------------------------------------ parameters

    @property
    def has_parameters(self):
        return bool(cell_has_parameters(self._c))

    @property
    def parameters(self):
        """``(a, b, c, alpha, beta, gamma)`` in angstroms and degrees."""
        cdef double a, b, c, al, be, ga
        if cell_get_parameters(self._c, &a, &b, &c, &al, &be, &ga):
            raise CrystFELError("Unit cell has no parameters")
        return (a * 1e10, b * 1e10, c * 1e10,
                al / _RAD, be / _RAD, ga / _RAD)

    @property
    def a(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    @property
    def c(self):
        return self.parameters[2]

    @property
    def alpha(self):
        return self.parameters[3]

    @property
    def beta(self):
        return self.parameters[4]

    @property
    def gamma(self):
        return self.parameters[5]

    @property
    def volume(self):
        """Cell volume in cubic angstroms."""
        return cell_get_volume(self._c) * 1e30

    @property
    def direct_axes(self):
        """The direct-space axes as a 3x3 array of angstroms, one row per
        axis (a, b, c)."""
        cdef double ax, ay, az, bx, by, bz, cx, cy, cz
        if cell_get_cartesian(self._c, &ax, &ay, &az, &bx, &by, &bz,
                              &cx, &cy, &cz):
            raise CrystFELError("Unit cell has no axes")
        return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]) * 1e10

    @property
    def reciprocal_axes(self):
        """The reciprocal-space axes as a 3x3 array of inverse angstroms, one
        row per axis (a*, b*, c*)."""
        cdef double ax, ay, az, bx, by, bz, cx, cy, cz
        if cell_get_reciprocal(self._c, &ax, &ay, &az, &bx, &by, &bz,
                               &cx, &cy, &cz):
            raise CrystFELError("Unit cell has no axes")
        return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]) * 1e-10

    # ------------------------------------------------------------- symmetry

    @property
    def centering(self):
        return chr(cell_get_centering(self._c))

    @centering.setter
    def centering(self, value):
        cell_set_centering(self._c, _one_char(value, "centering"))

    @property
    def lattice_type(self):
        return str_lattice(cell_get_lattice_type(self._c)).decode("utf-8")

    @lattice_type.setter
    def lattice_type(self, value):
        cell_set_lattice_type(self._c, _lattice_from_str(value))

    @property
    def unique_axis(self):
        return chr(cell_get_unique_axis(self._c))

    @unique_axis.setter
    def unique_axis(self, value):
        cell_set_unique_axis(self._c, _one_char(value, "unique_axis"))

    @property
    def right_handed(self):
        return bool(right_handed(self._c))

    @property
    def is_sensible(self):
        """True if the parameters are physically plausible (no zero or
        negative lengths, angles that close, ...)."""
        return bool(cell_is_sensible(self._c))

    def validate(self):
        """Raise :class:`CrystFELError` if the cell is not self-consistent.

        This is the same check CrystFEL applies to a cell file given on the
        command line: it verifies that the parameters agree with the stated
        centering, lattice type and unique axis.
        """
        if validate_cell(self._c):
            raise CrystFELError(_log.explain("Unit cell is not valid"))

    @property
    def bravais_lattice(self):
        """True if the combination of lattice type, centering and unique axis
        is one of the 14 Bravais lattices."""
        return bool(bravais_lattice(self._c))

    # ------------------------------------------------------- transformations

    def resolution(self, int h, int k, int l):
        """sin(theta)/lambda = 1/(2d) for the reflection *h,k,l*, in inverse
        angstroms.

        This is CrystFEL's own ``resolution()``.  Note the factor of two: for
        1/d, double it, or use :meth:`d_spacing`.
        """
        return resolution(self._c, h, k, l) * 1e-10

    def d_spacing(self, int h, int k, int l):
        """The d-spacing of the reflection *h,k,l*, in angstroms."""
        cdef double r = resolution(self._c, h, k, l)
        if r == 0.0:
            return float("inf")
        return 1e10 / (2.0 * r)

    def transform(self, matrix):
        """Apply an integer 3x3 transformation to the cell axes.

        The matrix is applied in the same sense as ``cell_tool --transform``.
        """
        cdef IntegerMatrix *m
        cdef c_UnitCell *out
        arr = np.asarray(matrix)
        if arr.shape != (3, 3):
            raise ValueError("Transformation matrix must be 3x3")
        ints = np.rint(arr).astype(np.int64)
        if not np.allclose(arr, ints):
            raise ValueError("Transformation matrix must have integer entries")
        m = intmat_create_3x3(ints[0, 0], ints[0, 1], ints[0, 2],
                              ints[1, 0], ints[1, 1], ints[1, 2],
                              ints[2, 0], ints[2, 1], ints[2, 2])
        if m is NULL:
            raise CrystFELError("Could not create transformation matrix")
        try:
            out = cell_transform_intmat(self._c, m)
        finally:
            intmat_free(m)
        if out is NULL:
            raise CrystFELError(_log.explain("Could not transform cell"))
        return wrap_cell(out, True, None)

    def uncenter(self):
        """Return ``(primitive_cell, C)``.

        *primitive_cell* is the primitive version of this cell, and *C* is the
        integer 3x3 matrix which was applied to the axes to get there.
        """
        cdef IntegerMatrix *C = NULL
        cdef c_UnitCell *out
        out = uncenter_cell(self._c, &C, NULL)
        if out is NULL:
            raise CrystFELError(_log.explain("Could not uncenter cell"))
        mat = np.empty((3, 3), dtype=np.int64)
        if C is not NULL:
            for i in range(3):
                for j in range(3):
                    mat[i, j] = intmat_get(C, i, j)
            intmat_free(C)
        return wrap_cell(out, True, None), mat

    def impose_bravais(self, lattice_type, unique_axis='*'):
        """Return a copy with the constraints of *lattice_type* imposed."""
        cdef c_UnitCell *out
        out = impose_bravais(self._c, _lattice_from_str(lattice_type),
                             _one_char(unique_axis, "unique_axis"))
        if out is NULL:
            raise CrystFELError(_log.explain("Could not impose lattice type"))
        return wrap_cell(out, True, None)

    # ----------------------------------------------------------- comparison

    def matches(self, UnitCell reference not None,
                tolerances=(0.05, 0.05, 0.05, 1.5, 1.5, 1.5)):
        """True if this cell matches *reference* within *tolerances*.

        The first three tolerances are fractional (on a, b, c), the last three
        are angles in **degrees**.  This is the same convention as
        ``indexamajig --tolerance``.
        """
        cdef double tols[6]
        _fill_tolerances(tolerances, tols)
        return bool(compare_cell_parameters(self._c, reference._c, tols))

    def reindexed_to(self, UnitCell reference not None,
                     tolerances=(0.05, 0.05, 0.05, 1.5, 1.5, 1.5)):
        """Look for a reindexed version of this cell which matches *reference*.

        Returns ``(cell, matrix)``, or ``(None, None)`` if there is no match.
        *matrix* is the 3x3 reindexing transformation, as floats (it can have
        fractional entries).
        """
        cdef double tols[6]
        cdef RationalMatrix *mb = NULL
        cdef c_UnitCell *out
        _fill_tolerances(tolerances, tols)
        out = compare_reindexed_cell_parameters(self._c, reference._c, tols,
                                                &mb)
        if out is NULL:
            if mb is not NULL:
                rtnl_mtx_free(mb)
            return None, None
        mat = None
        if mb is not NULL:
            mat = np.empty((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    mat[i, j] = rtnl_as_double(rtnl_mtx_get(mb, i, j))
            rtnl_mtx_free(mb)
        return wrap_cell(out, True, None), mat

    # ------------------------------------------------------------- niceties

    def copy(self):
        """A deep copy, preserving orientation as well as parameters."""
        cdef c_UnitCell *out = cell_new_from_cell(self._c)
        if out is NULL:
            raise CrystFELError(_log.explain("Could not copy cell"))
        return wrap_cell(out, True, None)

    def __repr__(self):
        if not self.has_parameters:
            return ("<UnitCell %s %s (no parameters)>"
                    % (self.lattice_type, self.centering))
        a, b, c, al, be, ga = self.parameters
        return ("<UnitCell %s %s %.2f %.2f %.2f A  %.2f %.2f %.2f deg>"
                % (self.lattice_type, self.centering, a, b, c, al, be, ga))
