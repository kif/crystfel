"""Exceptions raised by the CrystFEL bindings."""


class CrystFELError(Exception):
    """An operation in libcrystfel failed.

    libcrystfel reports failures by returning NULL (or non-zero) after writing
    an explanatory message through its own logging system.  The bindings
    capture that message (see :mod:`crystfel._log`) and attach it here, so the
    exception text is the same thing you would have seen on stderr from
    ``indexamajig``.
    """


class GeometryError(CrystFELError):
    """A geometry file could not be read, written or transformed."""


class IndexingError(CrystFELError):
    """The indexing system could not be set up or run."""
