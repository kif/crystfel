# Extension modules.
#
# Cython works out the full dotted module name by walking up from the .pyx
# file looking for __init__.py, so these compile as "crystfel.cell" and so on,
# which is what cross-module cimport needs.  The -I below puts src/ on
# Cython's include path so that "from crystfel.cell cimport UnitCell"
# resolves to src/crystfel/cell.pxd.

cython_args = ['-I' + meson.current_source_dir() / '..']

extension_modules = ['_log', 'cell', 'geometry', 'image', 'indexing']

foreach name : extension_modules
  py.extension_module(name,
                      name + '.pyx',
                      dependencies: [crystfeldep],
                      cython_args: cython_args,
                      install: true,
                      subdir: 'crystfel')
endforeach

py.install_sources(['__init__.py',
                    'errors.py'],
                   subdir: 'crystfel')

# The .pxd files are installed so that other Cython projects can cimport
# these classes and work with the C pointers directly.
py.install_sources(['_libcrystfel.pxd',
                    'cell.pxd',
                    'geometry.pxd',
                    'image.pxd'],
                   subdir: 'crystfel')
