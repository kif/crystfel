# cython: language_level=3

from crystfel._libcrystfel cimport (Crystal as c_Crystal, ImageFeatureList,
                                    image as c_image)


cdef class Image:
    cdef c_image *_c
    cdef object _dtempl

    cdef ImageFeatureList *_features(self) except NULL


cdef class Crystal:
    cdef c_Crystal *_c
    cdef object _parent
