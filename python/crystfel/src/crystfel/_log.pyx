# cython: language_level=3
"""Routing of libcrystfel's messages into the standard :mod:`logging` module.

libcrystfel writes its diagnostics with ``STATUS()`` and ``ERROR()``, which go
to stderr by default.  That is useless in a notebook or inside a worker
process, and it throws away the explanation of every failure, since the C API
only tells us "NULL" afterwards.

:func:`install` redirects both streams to the ``crystfel`` logger:

* ``STATUS`` messages become ``logging.INFO``
* ``ERROR`` messages become ``logging.ERROR``, and the most recent one is kept
  in thread-local storage so that the wrapper which detects the failure can put
  it into the exception it raises.

The callback is invoked from C, possibly from a thread which is running
indexing without the GIL, so it is declared ``with gil`` and must never let an
exception escape.
"""

import logging
import threading

from crystfel._libcrystfel cimport (LOG_MSG_ERROR, LogMsgFunc, log_msg_type,
                                    set_log_message_func)

logger = logging.getLogger("crystfel")

_tls = threading.local()

cdef bint _installed = False


cdef void _log_callback(log_msg_type msg_type, const char *msg,
                        void *vp) noexcept with gil:
    try:
        text = (<bytes> msg).decode("utf-8", "replace").rstrip("\n")
        if not text:
            return
        if msg_type == LOG_MSG_ERROR:
            _tls.last_error = text
            logger.error("%s", text)
        else:
            logger.info("%s", text)
    except BaseException:
        # A C callback must not propagate anything.  Losing a log line is
        # preferable to unwinding through libcrystfel.
        pass


def install():
    """Install the logging hook.  Idempotent; called when :mod:`crystfel` is
    imported."""
    global _installed
    if _installed:
        return
    set_log_message_func(<LogMsgFunc> _log_callback, NULL)
    _installed = True


def take_last_error():
    """Return and clear the last ``ERROR`` message from this thread.

    Returns ``None`` if libcrystfel has not complained since the last call.
    """
    msg = getattr(_tls, "last_error", None)
    _tls.last_error = None
    return msg


def explain(fallback):
    """Combine *fallback* with libcrystfel's own explanation, if there is one."""
    detail = take_last_error()
    if detail is None:
        return fallback
    return "%s: %s" % (fallback, detail)
