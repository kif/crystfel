# cython: language_level=3
#
# C declarations for libcrystfel.
#
# This file contains no code.  Every declaration below is checked against the
# *installed* libcrystfel headers when Cython's output is compiled, so the
# layout of the public structures (struct image, struct detgeom_panel, ...)
# always matches the library we are built against.  A mismatch is a compile
# error rather than a silent memory corruption, which is the main reason these
# bindings use Cython rather than ctypes.
#
# Deliberately declared *opaque* here: the per-engine indexing option
# structures (struct taketwo_options, struct xgandalf_options, ...).  Their
# real definition depends on HAVE_* macros which come from
# libcrystfel-config.h, and that header is not installed.  Never allocate or
# dereference them: obtain them from default_method_options(), which allocates
# them inside the library, and pass them straight to setup_indexing().

from libc.stdio cimport FILE


# ----------------------------------------------------------------- utils.h --

cdef extern from "utils.h" nogil:

    cdef enum log_msg_type:
        LOG_MSG_STATUS
        LOG_MSG_ERROR

    ctypedef void (*LogMsgFunc)(log_msg_type msg_type, const char *msg,
                                void *vp) noexcept nogil

    void set_log_message_func(LogMsgFunc new_log_msg_func, void *vp)

    void cffree(void *ptr)


cdef extern from *:
    """
    /* utils.h defines likely()/unlikely() as public macros, which collide
     * with Cython's own definitions of the same names.  The two are
     * identical in effect, but the redefinition is a warning on every
     * compile, so drop CrystFEL's and let Cython's stand. */
    #undef likely
    #undef unlikely
    """
    pass


# -------------------------------------------------------- integer_matrix.h --

cdef extern from "integer_matrix.h" nogil:

    ctypedef struct IntegerMatrix:
        pass

    void intmat_free(IntegerMatrix *m)
    void intmat_size(const IntegerMatrix *m, unsigned int *rows,
                     unsigned int *cols)
    signed int intmat_get(const IntegerMatrix *m, unsigned int i,
                          unsigned int j)
    IntegerMatrix *intmat_create_3x3(signed int m11, signed int m12,
                                     signed int m13,
                                     signed int m21, signed int m22,
                                     signed int m23,
                                     signed int m31, signed int m32,
                                     signed int m33)


# -------------------------------------------------------------- rational.h --

cdef extern from "rational.h" nogil:

    ctypedef struct Rational:
        pass

    ctypedef struct RationalMatrix:
        pass

    double rtnl_as_double(Rational r)
    Rational rtnl_mtx_get(const RationalMatrix *m, int i, int j)
    void rtnl_mtx_free(RationalMatrix *mtx)


# ------------------------------------------------------------------ cell.h --

cdef extern from "cell.h" nogil:

    ctypedef struct UnitCell:
        pass

    ctypedef enum LatticeType:
        L_TRICLINIC
        L_MONOCLINIC
        L_ORTHORHOMBIC
        L_TETRAGONAL
        L_RHOMBOHEDRAL
        L_HEXAGONAL
        L_CUBIC

    UnitCell *cell_new()
    UnitCell *cell_new_from_cell(const UnitCell *orig)
    void cell_free(UnitCell *cell)

    UnitCell *cell_new_from_parameters(double a, double b, double c,
                                       double alpha, double beta, double gamma)

    int cell_has_parameters(const UnitCell *cell)

    void cell_set_parameters(UnitCell *cell, double a, double b, double c,
                             double alpha, double beta, double gamma)
    int cell_get_parameters(UnitCell *cell, double *a, double *b, double *c,
                            double *alpha, double *beta, double *gamma)

    void cell_set_cartesian(UnitCell *cell,
                            double ax, double ay, double az,
                            double bx, double by, double bz,
                            double cx, double cy, double cz)
    int cell_get_cartesian(UnitCell *cell,
                           double *ax, double *ay, double *az,
                           double *bx, double *by, double *bz,
                           double *cx, double *cy, double *cz)

    void cell_set_reciprocal(UnitCell *cell,
                             double asx, double asy, double asz,
                             double bsx, double bsy, double bsz,
                             double csx, double csy, double csz)
    int cell_get_reciprocal(UnitCell *cell,
                            double *asx, double *asy, double *asz,
                            double *bsx, double *bsy, double *bsz,
                            double *csx, double *csy, double *csz)

    LatticeType cell_get_lattice_type(const UnitCell *cell)
    void cell_set_lattice_type(UnitCell *cell, LatticeType lattice_type)

    char cell_get_centering(const UnitCell *cell)
    void cell_set_centering(UnitCell *cell, char centering)

    char cell_get_unique_axis(const UnitCell *cell)
    void cell_set_unique_axis(UnitCell *cell, char unique_axis)

    UnitCell *cell_transform_intmat(UnitCell *cell, IntegerMatrix *m)
    UnitCell *cell_transform_intmat_inverse(UnitCell *cell, IntegerMatrix *m)
    UnitCell *cell_transform_rational(UnitCell *cell, RationalMatrix *m)
    UnitCell *cell_transform_rational_inverse(UnitCell *cell, RationalMatrix *m)


# ------------------------------------------------------------ cell-utils.h --

cdef extern from "cell-utils.h" nogil:

    double resolution(UnitCell *cell, signed int h, signed int k, signed int l)
    double lowest_reflection(UnitCell *cell)

    UnitCell *load_cell_from_file(const char *filename)
    int write_cell_to_file(UnitCell *cell, const char *filename)

    void cell_print(UnitCell *cell)

    int cell_is_sensible(UnitCell *cell)
    int validate_cell(UnitCell *cell)
    int bravais_lattice(UnitCell *cell)
    int right_handed(UnitCell *cell)
    double cell_get_volume(UnitCell *cell)

    const char *str_lattice(LatticeType l)
    LatticeType lattice_from_str(const char *s)

    UnitCell *uncenter_cell(UnitCell *cell, IntegerMatrix **pC,
                            RationalMatrix **pCi)
    UnitCell *impose_bravais(UnitCell *cell, LatticeType latt, char ua)

    int compare_cell_parameters(UnitCell *cell, UnitCell *reference,
                                const double *tols)
    int compare_cell_parameters_and_orientation(UnitCell *cell,
                                                UnitCell *reference,
                                                const double *tols)
    UnitCell *compare_reindexed_cell_parameters(UnitCell *cell_in,
                                                UnitCell *reference_in,
                                                const double *tols,
                                                RationalMatrix **pmb)


# --------------------------------------------------------------- detgeom.h --

cdef extern from "detgeom.h" nogil:

    cdef struct detgeom_panel_group:
        char *name
        int n_children
        detgeom_panel_group *parent
        int serial
        double cx, cy, cz
        detgeom_panel_group **children
        # Really "struct detgeom_panel *", declared as void * to avoid a
        # circular reference between the two struct declarations.
        void *panel

    cdef struct detgeom_panel:
        char *name
        double cnx, cny, cnz
        double pixel_pitch
        double adu_per_photon
        double max_adu
        double fsx, fsy, fsz
        double ssx, ssy, ssz
        int w, h
        const detgeom_panel_group *group

    cdef struct detgeom:
        detgeom_panel *panels
        int n_panels
        detgeom_panel_group *top_group

    void detgeom_transform_coords(detgeom_panel *p, double fs, double ss,
                                  double wavelength, double dx, double dy,
                                  double *r)
    void detgeom_free(detgeom *dg)
    double detgeom_max_resolution(detgeom *dg, double wavelength)
    double detgeom_mean_camera_length(detgeom *dg)
    detgeom_panel *detgeom_find_panel(detgeom *dg, const char *name)
    void detgeom_show_hierarchy(const detgeom *dg)
    void detgeom_translate_detector_m(detgeom *dg, double x, double y,
                                      double z)
    int detgeom_group_center(const detgeom_panel_group *grp,
                             double *x, double *y, double *z)


# ---------------------------------------------------------- datatemplate.h --

cdef extern from "datatemplate.h" nogil:

    ctypedef struct DataTemplate:
        pass

    cdef struct dg_group_info:
        const char *name
        int serial
        int hierarchy_level
        int leaf

    DataTemplate *data_template_new_from_file(const char *filename)
    DataTemplate *data_template_new_from_string(const char *string_in)
    void data_template_free(DataTemplate *dt)

    const char *data_template_panel_number_to_name(const DataTemplate *dt,
                                                   int pn)
    int data_template_panel_name_to_number(const DataTemplate *dt,
                                           const char *panel_name, int *pn)

    int data_template_file_to_panel_coords(const DataTemplate *dt,
                                           float *pfs, float *pss, int pn)
    int data_template_slabby_file_to_panel_coords(const DataTemplate *dt,
                                                  float *pfs, float *pss,
                                                  int *ppn)
    int data_template_panel_to_file_coords(const DataTemplate *dt, int pn,
                                           float *pfs, float *pss)

    int data_template_get_slab_extents(const DataTemplate *dt, int *pw,
                                       int *ph)
    double data_template_get_wavelength_if_possible(const DataTemplate *dt)
    double data_template_get_clen_if_possible(const DataTemplate *dt)
    detgeom *data_template_get_detgeom_if_possible(const DataTemplate *dt,
                                                   int no_clen_ok)

    void data_template_show_hierarchy(const DataTemplate *dtempl)
    dg_group_info *data_template_group_info(const DataTemplate *dtempl, int *n)

    int data_template_translate_group_px(DataTemplate *dtempl,
                                         const char *group_name,
                                         double x, double y, double z)
    int data_template_translate_group_m(DataTemplate *dtempl,
                                        const char *group_name,
                                        double x, double y, double z)
    int data_template_rotate_group(DataTemplate *dtempl,
                                   const char *group_name,
                                   double ang, char axis)
    int data_template_add_to_scanv_beamxy(DataTemplate *dtempl,
                                          double *updates)

    void data_template_reset_total_movements(DataTemplate *dtempl)
    void data_template_print_total_movements(const DataTemplate *dtempl)

    int data_template_write_to_fh(const DataTemplate *dtempl, FILE *fh)
    int data_template_write_to_file(const DataTemplate *dtempl,
                                    const char *filename)


# --------------------------------------------------------------- crystal.h --

cdef extern from "crystal.h" nogil:

    ctypedef struct Crystal:
        pass

    void crystal_free(Crystal *cryst)
    UnitCell *crystal_get_cell(Crystal *cryst)
    double crystal_get_profile_radius(const Crystal *cryst)
    double crystal_get_resolution_limit(Crystal *cryst)
    double crystal_get_mosaicity(Crystal *cryst)
    int crystal_get_user_flag(Crystal *cryst)
    const char *crystal_get_notes(Crystal *cryst)
    void crystal_get_det_shift(Crystal *cryst, double *shift_x,
                               double *shift_y)


# ------------------------------- index.h, part 1: types needed by image.h ---

cdef extern from "index.h" nogil:

    ctypedef enum IndexingMethod:
        INDEXING_NONE
        INDEXING_DIRAX
        INDEXING_MOSFLM
        INDEXING_FELIX
        INDEXING_XDS
        INDEXING_SIMULATION
        INDEXING_FILE
        INDEXING_ASDF
        INDEXING_TAKETWO
        INDEXING_XGANDALF
        INDEXING_PINKINDEXER
        INDEXING_FFBIDX
        INDEXING_SMALLCELL
        INDEXING_ERROR

    ctypedef enum IndexingFlags:
        INDEXING_RETRY
        INDEXING_MULTI
        INDEXING_REFINE
        INDEXING_CHECK_PEAKS
        INDEXING_CHECK_CELL


# ----------------------------------------------------------------- image.h --

cdef extern from "image.h" nogil:

    ctypedef struct ImageFeatureList:
        pass

    ctypedef struct ImageDataArrays:
        pass

    ctypedef struct RefList:
        pass

    ctypedef struct Spectrum:
        pass

    cdef struct imagefeature:
        double fs
        double ss
        int pn
        double intensity
        const char *name

    cdef struct crystal_refls:
        Crystal *cr
        RefList *refls
        int image_owns_crystal
        int image_owns_refls

    cdef struct image:
        float **dp
        int **bad
        float **sat
        int hit
        crystal_refls *crystals
        int n_crystals
        IndexingMethod indexed_by
        int n_indexing_tries
        # "detgeom" clashes with the struct tag, and "lambda" is a Python
        # keyword, so both are renamed on the Cython side only.
        detgeom *det "detgeom"
        char *filename
        char *ev
        int serial
        double wavelength "lambda"
        double div
        double bw
        double peak_resolution
        ImageFeatureList *features
        ImageDataArrays *ida
        int owns_peaklist

    ImageFeatureList *image_feature_list_new()
    void image_feature_list_free(ImageFeatureList *flist)
    void image_add_feature(ImageFeatureList *flist, double fs, double ss,
                           int pn, double intensity, const char *name)
    int image_feature_count(ImageFeatureList *flist)
    imagefeature *image_get_feature(ImageFeatureList *flist, int idx)

    image *image_new()
    image *image_create_for_simulation(const DataTemplate *dtempl)
    void image_free(image *img)


# ----------------------------- index.h, part 2: the rest of the interface ---

cdef extern from "index.h" nogil:

    ctypedef struct IndexingPrivate:
        pass

    # Opaque on purpose: see the note at the top of this file.
    cdef struct taketwo_options:
        pass
    cdef struct xgandalf_options:
        pass
    cdef struct ffbidx_options:
        pass
    cdef struct pinkindexer_options:
        pass
    cdef struct felix_options:
        pass
    cdef struct fromfile_options:
        pass
    cdef struct smallcell_options:
        pass
    cdef struct asdf_options:
        pass

    const char *indexer_str(IndexingMethod indm)
    IndexingMethod get_indm_from_string_2(const char *method, int *err)

    void default_method_options(taketwo_options **ttopts,
                                xgandalf_options **xgandalf_opts,
                                ffbidx_options **ffbidx_opts,
                                pinkindexer_options **pinkIndexer_opts,
                                felix_options **felix_opts,
                                fromfile_options **fromfile_opts,
                                smallcell_options **smallcell_opts,
                                asdf_options **asdf_opts)

    IndexingPrivate *setup_indexing(const char *methods, UnitCell *cell,
                                    float *ltl, IndexingFlags flags,
                                    double wavelength_estimate,
                                    double clen_estimate, int n_threads,
                                    taketwo_options *ttopts,
                                    xgandalf_options *xgandalf_opts,
                                    ffbidx_options *ffbidx_opts,
                                    pinkindexer_options *pinkIndexer_opts,
                                    felix_options *felix_opts,
                                    fromfile_options *fromfile_opts,
                                    smallcell_options *smallcell_opts,
                                    asdf_options *asdf_opts)

    const IndexingMethod *indexing_methods(IndexingPrivate *p, int *n)
    char *detect_indexing_methods(UnitCell *cell)
    void print_indexing_info(IndexingPrivate *ipriv)
    void index_pattern(image *img, IndexingPrivate *ipriv)
    void cleanup_indexing(IndexingPrivate *ipriv)
