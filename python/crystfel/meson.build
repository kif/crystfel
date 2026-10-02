# Python bindings for libcrystfel
#
# This is a *separate* Meson project from CrystFEL itself, because meson-python
# requires the Python package to be the root of its own Meson project.  It
# builds against an already-installed libcrystfel, found via pkg-config.
#
# To build against an uninstalled CrystFEL, point PKG_CONFIG_PATH at the
# pkgconfig directory of your CrystFEL installation prefix, e.g.
#
#   meson install -C ../../build --destdir /tmp/cf
#   PKG_CONFIG_PATH=/tmp/cf/usr/local/lib64/pkgconfig pip install .

project('crystfel-python', 'c', 'cython',
        version: '0.13.0',
        license: 'GPL-3.0-or-later',
        meson_version: '>= 1.1.0',
        default_options: ['buildtype=release'])

py = import('python').find_installation(pure: false)

# The API version of libcrystfel is the soversion of the shared library, and is
# bumped whenever the ABI changes.  The declarations in _libcrystfel.pxd are
# checked against the real headers at compile time, but the *semantics* are
# only guaranteed for the versions we have tested against.
crystfeldep = dependency('crystfel', version: '>= 0.13.0', required: false)

if not crystfeldep.found()
  error('''Could not find libcrystfel via pkg-config.

  Install CrystFEL first, then make its pkg-config file visible, e.g.:

      export PKG_CONFIG_PATH=/path/to/crystfel/lib64/pkgconfig

  See INSTALL.md in the top level of the CrystFEL source tree.''')
endif

subdir('src/crystfel')
