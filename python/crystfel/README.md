# CrystFEL Python bindings

Cython bindings to `libcrystfel`, covering detector geometry and indexing.

These are *thin*: each Python object holds a pointer to the corresponding C
structure and frees it when collected.  The C declarations in
`src/crystfel/_libcrystfel.pxd` are checked against the **installed CrystFEL
headers** when the extension is compiled, so there is no hand-maintained copy
of any structure layout to drift out of date.  If `struct image` changes
upstream, this package fails to compile rather than corrupting memory.


## Building

This is a separate Meson project from CrystFEL itself, because `meson-python`
requires the Python package to be the root of its own Meson project.  It
builds against an **already-installed** libcrystfel, found with `pkg-config`.

```sh
# 1. Build and install CrystFEL itself (see ../../INSTALL.md)
meson setup build --prefix=$HOME/crystfel
meson install -C build

# 2. Build the bindings against it
export PKG_CONFIG_PATH=$HOME/crystfel/lib64/pkgconfig   # or lib/pkgconfig
pip install ./python/crystfel
```

If `libcrystfel` is not installed in a system location, the dynamic linker
also needs to find it at run time (`LD_LIBRARY_PATH`, or a prefix whose
`lib64` is already configured).

For development, an editable install rebuilds the extensions automatically
whenever a `.pyx` file changes:

```sh
pip install --no-build-isolation -e ./python/crystfel
```


## Quick tour

```python
import crystfel

dt = crystfel.DataTemplate('detector.geom')

# --- geometry transformations (what adjust_detector / align_detector do) ---
for g in dt.groups:
    print('  ' * g.hierarchy_level, g.name)

dt.translate_group('q0', 0.0, 0.0, 100e-6)   # 100 um along z, in metres
dt.rotate_group('q0', 0.05, 'z')             # 0.05 degrees about z
dt.write('detector-moved.geom')

# --- coordinate systems ---
det = dt.detector()
panel = det['q0a0']
q = det.transform_coords(panel, fs, ss, dt.wavelength)   # (N, 3) in m^-1

# --- indexing ---
cell = crystfel.UnitCell(63.0, 63.0, 42.0, 90, 90, 120,
                         lattice_type='hexagonal', unique_axis='c')
indexer = crystfel.Indexer('xgandalf', dt, cell)

img = crystfel.Image(dt)
img.set_peaks(fs, ss, pn, intensity)
if indexer.index(img):
    print(img.crystals[0].cell.parameters)
```


## Things to know

**Units are not uniform, because CrystFEL's are not.**
`crystfel.cell` uses angstroms and degrees, like a `.cell` file.
`crystfel.geometry` and `crystfel.image` use metres, inverse metres and
radians, like a `.geom` file -- the exception being `rotate_group()`, which
takes degrees because that is what you actually want to type.

**Peak coordinates are distances, not indices.**  `fs` and `ss` are measured
in pixels *from the corner of the panel*, so the centre of the first pixel is
at 0.5, 0.5.  Half-pixel errors here are a classic reason for indexing to stop
working with no error message.

**One `Indexer` per thread.**  libcrystfel keeps per-engine state in the
`IndexingPrivate`, and `indexamajig` gets its parallelism from forking one
worker process per core, not from threads.  `Indexer.index()` releases the
GIL, so a thread pool with one `Indexer` each works for the in-process engines
(taketwo, xgandalf, asdf, smallcell, pinkindexer, ffbidx).  The engines that
drive an external program (mosflm, dirax, xds, felix) fork through a
pseudo-terminal; use `multiprocessing` for those.

**libcrystfel's messages go to the `crystfel` logger.**  `STATUS` becomes
`INFO` and `ERROR` becomes `ERROR`, and the most recent error is attached to
the `CrystFELError` that the failure raises, so exceptions carry the same
explanation you would have seen on stderr.  Turn the running commentary on
with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

**The indexing option structures are opaque on purpose.**  `struct
ffbidx_options` changes size depending on whether the library was compiled
with `HAVE_FFBIDX`, and `libcrystfel-config.h` is not installed, so the
layout cannot be known from outside.  The bindings never allocate or
dereference them: they come from `default_method_options()` and go straight
back into `setup_indexing()`.  This is also why per-engine tuning parameters
are not yet exposed -- see below.


## Not covered yet

* Per-engine indexing options (xgandalf sampling pitch, taketwo tolerances,
  ...).  Exposing these needs small accessor functions in libcrystfel, since
  the structure layouts are not knowable from an installed header.
* Peak search.  `search_peaks()` and `search_peaks_peakfinder9()` are
  reachable from the installed `peaks.h`, but `peakfinder8()` is not:
  `peakfinder8.h` is missing from `install_headers` in
  `libcrystfel/meson.build`.
* Reading image data (`image_read`), reflection lists, integration, streams,
  merging and figures of merit.
