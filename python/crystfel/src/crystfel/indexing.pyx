# cython: language_level=3
"""The indexing system.

Threading
---------
An :class:`Indexer` owns per-engine state which libcrystfel reuses from one
pattern to the next (``struct _indexingprivate`` holds a ``void **
engine_private``), so **one Indexer must not be used from two threads at the
same time**.  This mirrors how ``indexamajig`` works: its parallelism is at the
*process* level -- it forks one worker per core and gives each its own indexing
setup.

With that rule respected, :meth:`Indexer.index` releases the GIL, so one
Indexer per thread in a :class:`~concurrent.futures.ThreadPoolExecutor` scales
properly for the in-process engines (``taketwo``, ``xgandalf``, ``asdf``,
``smallcell``, ``pinkindexer``, ``ffbidx``).

The engines which drive an external program (``mosflm``, ``dirax``, ``xds``,
``felix``) fork a child process through a pseudo-terminal.  Doing that from a
thread of a long-lived Python process is asking for trouble; use
:mod:`multiprocessing` for those.
"""

from libc.math cimport M_PI

from crystfel import _log
from crystfel.errors import IndexingError

from crystfel.cell cimport UnitCell
from crystfel.geometry cimport DataTemplate
from crystfel.image cimport Image

from crystfel._libcrystfel cimport (INDEXING_CHECK_CELL, INDEXING_CHECK_PEAKS,
                                    INDEXING_MULTI, INDEXING_REFINE,
                                    INDEXING_RETRY, IndexingFlags,
                                    IndexingMethod, IndexingPrivate,
                                    UnitCell as c_UnitCell, asdf_options,
                                    cffree, cleanup_indexing,
                                    default_method_options,
                                    detect_indexing_methods, felix_options,
                                    ffbidx_options, fromfile_options,
                                    image as c_image, index_pattern,
                                    indexer_str, indexing_methods,
                                    pinkindexer_options, print_indexing_info,
                                    setup_indexing, smallcell_options,
                                    taketwo_options, xgandalf_options)

cdef double _RAD = M_PI / 180.0


def detect_methods(UnitCell cell=None):
    """Return the indexing methods CrystFEL would choose by default.

    This is what ``indexamajig`` does when ``--indexing`` is not given.
    """
    cdef c_UnitCell *c = NULL if cell is None else cell._c
    cdef char *out = detect_indexing_methods(c)
    if out is NULL:
        return None
    try:
        return out.decode("utf-8")
    finally:
        cffree(out)


cdef class Indexer:
    """A configured indexing engine.

    >>> indexer = Indexer('xgandalf', dtempl, cell)
    >>> n = indexer.index(image)

    *methods* is the usual CrystFEL method string, for example
    ``'xgandalf'`` or ``'taketwo-nocell,asdf'``.

    *tolerances* are the six values of ``indexamajig --tolerance``: three
    fractional tolerances on the cell lengths, then three angle tolerances in
    **degrees**.

    *wavelength* and *camera_length* (both in metres) are only needed when the
    geometry does not pin them down -- when they are read from file headers,
    there is no file here to read them from, so you must say what to expect.
    """

    cdef IndexingPrivate *_c
    cdef taketwo_options *_tt
    cdef xgandalf_options *_xg
    cdef ffbidx_options *_ffb
    cdef pinkindexer_options *_pink
    cdef felix_options *_felix
    cdef fromfile_options *_fromfile
    cdef smallcell_options *_smallcell
    cdef asdf_options *_asdf
    cdef object _cell

    def __cinit__(self, *args, **kwargs):
        self._c = NULL
        self._tt = NULL
        self._xg = NULL
        self._ffb = NULL
        self._pink = NULL
        self._felix = NULL
        self._fromfile = NULL
        self._smallcell = NULL
        self._asdf = NULL
        self._cell = None

    def __init__(self, methods, DataTemplate dtempl not None,
                 UnitCell cell=None,
                 tolerances=(0.05, 0.05, 0.05, 1.5, 1.5, 1.5),
                 retry=True, multi_lattice=False, refine=True,
                 check_peaks=True, check_cell=True,
                 wavelength=None, camera_length=None, int n_threads=1):

        cdef bytes raw = str(methods).encode("utf-8")
        cdef float ltl[6]
        cdef int flags = 0
        cdef c_UnitCell *ccell = NULL

        if wavelength is None:
            wavelength = dtempl.wavelength
            if wavelength is None:
                raise IndexingError(
                    "The geometry does not give the wavelength (it comes from "
                    "a file header), so you must pass wavelength=... in metres")
        if camera_length is None:
            camera_length = dtempl.camera_length
            if camera_length is None:
                raise IndexingError(
                    "The geometry does not give the camera length (it comes "
                    "from a file header), so you must pass "
                    "camera_length=... in metres")

        if len(tolerances) != 6:
            raise ValueError("Need exactly six tolerances")
        for i in range(3):
            ltl[i] = float(tolerances[i])
        for i in range(3, 6):
            ltl[i] = float(tolerances[i]) * _RAD

        if retry:
            flags |= <int> INDEXING_RETRY
        if multi_lattice:
            flags |= <int> INDEXING_MULTI
        if refine:
            flags |= <int> INDEXING_REFINE
        if check_peaks:
            flags |= <int> INDEXING_CHECK_PEAKS
        if check_cell:
            flags |= <int> INDEXING_CHECK_CELL

        if cell is not None:
            ccell = cell._c
            self._cell = cell          # keep the cell alive

        # The option structures are allocated inside libcrystfel, because
        # their layout depends on how the library was compiled.  The engines
        # keep the pointers (taketwo_prepare() does "tp->opts = opts"), so
        # they must stay alive for as long as the IndexingPrivate does.
        default_method_options(&self._tt, &self._xg, &self._ffb, &self._pink,
                               &self._felix, &self._fromfile,
                               &self._smallcell, &self._asdf)

        self._c = setup_indexing(raw, ccell, ltl, <IndexingFlags> flags,
                                 float(wavelength), float(camera_length),
                                 n_threads,
                                 self._tt, self._xg, self._ffb, self._pink,
                                 self._felix, self._fromfile,
                                 self._smallcell, self._asdf)
        if self._c is NULL:
            raise IndexingError(_log.explain("Could not set up indexing with "
                                             "method(s) %r" % methods))

    def __dealloc__(self):
        if self._c is not NULL:
            cleanup_indexing(self._c)
            self._c = NULL
        # Free the option structures only after cleanup_indexing(), which
        # tears down the engines still pointing at them.
        cffree(self._tt)
        cffree(self._xg)
        cffree(self._ffb)
        cffree(self._pink)
        cffree(self._felix)
        cffree(self._fromfile)
        cffree(self._smallcell)
        cffree(self._asdf)

    @property
    def methods(self):
        """The indexing methods actually set up, in the order they are tried."""
        cdef int n = 0
        cdef const IndexingMethod *m = indexing_methods(self._c, &n)
        if m is NULL:
            return ()
        return tuple(indexer_str(m[i]).decode("utf-8") for i in range(n))

    def show_info(self):
        """Print the configuration, as ``indexamajig`` does at startup."""
        print_indexing_info(self._c)

    def index(self, Image image not None):
        """Index one pattern, in place.

        The crystals found are appended to ``image.crystals``; the return
        value is the number of crystals in the image afterwards.

        The GIL is released for the duration.  See the module docstring: do
        not call this concurrently on one Indexer.
        """
        cdef c_image *img = image._c
        cdef IndexingPrivate *ipriv = self._c
        if image.n_peaks == 0:
            raise IndexingError("The image has no peaks to index")
        with nogil:
            index_pattern(img, ipriv)
        return image.n_crystals

    def __repr__(self):
        return "<Indexer %s>" % ", ".join(self.methods)
