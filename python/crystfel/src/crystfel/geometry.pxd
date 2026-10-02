# cython: language_level=3

from crystfel._libcrystfel cimport (DataTemplate as c_DataTemplate,
                                    detgeom as c_detgeom,
                                    detgeom_panel as c_detgeom_panel)


cdef class DataTemplate:
    cdef c_DataTemplate *_c


cdef class Detector:
    cdef c_detgeom *_c
    cdef bint _owned
    cdef object _parent


cdef class Panel:
    cdef c_detgeom_panel *_c
    cdef Detector _det


cdef Detector wrap_detector(c_detgeom *dg, bint owned, object parent)
