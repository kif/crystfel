# cython: language_level=3

from crystfel._libcrystfel cimport UnitCell as c_UnitCell


cdef class UnitCell:
    cdef c_UnitCell *_c
    cdef bint _owned
    cdef object _parent


cdef UnitCell wrap_cell(c_UnitCell *c, bint owned, object parent)
