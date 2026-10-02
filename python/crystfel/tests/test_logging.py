import logging

import pytest

import crystfel


def test_errors_reach_the_logger(caplog):
    with caplog.at_level(logging.ERROR, logger='crystfel'):
        with pytest.raises(crystfel.GeometryError):
            crystfel.DataTemplate('/nonexistent/detector.geom')
    assert any('crystfel' == r.name for r in caplog.records)


def test_exception_carries_libcrystfel_explanation():
    """The message libcrystfel wrote must end up in the exception, not just
    on stderr."""
    try:
        crystfel.DataTemplate('/nonexistent/detector.geom')
    except crystfel.GeometryError as exc:
        text = str(exc)
    else:
        pytest.fail('expected GeometryError')

    assert '/nonexistent/detector.geom' in text
    # libcrystfel's own wording is appended after a colon
    assert ':' in text


def test_status_messages_are_info_not_error(caplog, dtempl):
    with caplog.at_level(logging.INFO, logger='crystfel'):
        dtempl.show_hierarchy()
    assert caplog.records, 'show_hierarchy() produced no log output'
    assert all(r.levelno == logging.INFO for r in caplog.records)
