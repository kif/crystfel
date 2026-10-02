import numpy as np
import pytest

import crystfel
from conftest import synthetic_peaks

#: In-process engines, with how many of three synthetic patterns each is
#: expected to index.  taketwo is much more sensitive to the peak list than
#: the other two, so it only has to succeed once.
ENGINES = [('taketwo', 1), ('asdf', 3), ('xgandalf', 3)]


def available(name, dtempl, cell):
    try:
        crystfel.Indexer(name, dtempl, cell)
    except crystfel.IndexingError:
        return False
    return True


def test_setup_and_report(dtempl, cell):
    indexer = crystfel.Indexer('taketwo', dtempl, cell)
    assert indexer.methods == ('taketwo',)
    assert 'taketwo' in repr(indexer)


def test_unknown_method(dtempl, cell):
    with pytest.raises(crystfel.IndexingError):
        crystfel.Indexer('notanindexer', dtempl, cell)


def test_detect_methods(cell):
    methods = crystfel.detect_methods(cell)
    assert isinstance(methods, str)
    assert 'taketwo' in methods


def test_wavelength_must_be_known(cell):
    """A geometry which reads the wavelength from a header cannot be used
    without telling us what to expect."""
    from conftest import SIMPLE_GEOM
    geom = SIMPLE_GEOM.replace('wavelength = 1.0 A',
                               'wavelength = /LCLS/wavelength m')
    dt = crystfel.DataTemplate.from_string(geom)
    with pytest.raises(crystfel.IndexingError, match='wavelength'):
        crystfel.Indexer('taketwo', dt, cell)
    # ... but it works when we say
    crystfel.Indexer('taketwo', dt, cell, wavelength=1e-10)


def test_image_needs_a_literal_geometry(cell):
    from conftest import SIMPLE_GEOM
    geom = SIMPLE_GEOM.replace('wavelength = 1.0 A',
                               'wavelength = /LCLS/wavelength m')
    dt = crystfel.DataTemplate.from_string(geom)
    with pytest.raises(crystfel.CrystFELError):
        crystfel.Image(dt)


def test_image_peaks_round_trip(dtempl):
    img = crystfel.Image(dtempl)
    assert img.n_peaks == 0

    fs = np.array([10.5, 20.5, 30.5])
    ss = np.array([1.5, 2.5, 3.5])
    img.set_peaks(fs, ss, 0, [100.0, 200.0, 300.0])

    assert img.n_peaks == 3
    peaks = img.peaks
    assert peaks.dtype == crystfel.PEAK_DTYPE
    assert peaks['fs'] == pytest.approx(fs)
    assert peaks['ss'] == pytest.approx(ss)
    assert list(peaks['pn']) == [0, 0, 0]
    assert peaks['intensity'] == pytest.approx([100.0, 200.0, 300.0])


def test_image_peaks_broadcast(dtempl):
    img = crystfel.Image(dtempl)
    img.set_peaks(np.arange(5.0), np.arange(5.0), 0)
    assert img.n_peaks == 5
    assert set(img.peaks['pn']) == {0}


def test_add_and_clear_peaks(dtempl):
    img = crystfel.Image(dtempl)
    img.add_peak(1.5, 2.5, 0, 42.0)
    assert img.n_peaks == 1
    assert img.peaks['intensity'][0] == pytest.approx(42.0)
    img.clear_peaks()
    assert img.n_peaks == 0


def test_image_properties(dtempl):
    img = crystfel.Image(dtempl)
    assert img.wavelength == pytest.approx(1e-10)
    img.serial = 17
    assert img.serial == 17
    assert img.n_crystals == 0
    assert img.indexed_by == 'none'
    assert len(img.detector) == 1


def test_indexing_an_empty_image_is_refused(dtempl, cell):
    indexer = crystfel.Indexer('taketwo', dtempl, cell)
    img = crystfel.Image(dtempl)
    with pytest.raises(crystfel.IndexingError):
        indexer.index(img)


@pytest.mark.parametrize('method,min_indexed', ENGINES)
def test_index_synthetic_pattern(dtempl, cell, method, min_indexed):
    """Index peaks generated from a known cell, and get that cell back."""
    if not available(method, dtempl, cell):
        pytest.skip('%s is not available in this build of CrystFEL' % method)

    indexer = crystfel.Indexer(method, dtempl, cell)
    rng = np.random.default_rng(20260102)

    indexed = 0
    for _ in range(3):
        fs, ss = synthetic_peaks(dtempl, cell, rng)
        assert len(fs) > 50, 'the synthetic pattern is too sparse to index'
        img = crystfel.Image(dtempl)
        img.set_peaks(fs, ss, 0, 1.0)
        if indexer.index(img):
            indexed += 1
            found = img.crystals[0].cell
            assert found.matches(cell, tolerances=(0.05, 0.05, 0.05, 2, 2, 2))
            assert img.indexed_by == method
            assert img.crystals[0].det_shift == pytest.approx((0.0, 0.0),
                                                              abs=1e-3)
    assert indexed >= min_indexed


def test_crystals_keep_the_image_alive(dtempl, cell):
    """A Crystal is a view into memory owned by its Image, so holding one
    must keep the image from being collected."""
    indexer = crystfel.Indexer('xgandalf', dtempl, cell)
    rng = np.random.default_rng(5)
    fs, ss = synthetic_peaks(dtempl, cell, rng)
    img = crystfel.Image(dtempl)
    img.set_peaks(fs, ss, 0, 1.0)
    if not indexer.index(img):
        pytest.skip('this pattern did not index')

    crystal = img.crystals[0]
    found = crystal.cell
    del img
    import gc
    gc.collect()
    # Still usable: the crystal holds a reference to the image
    assert found.parameters[0] == pytest.approx(50.0, abs=2.0)


def test_one_indexer_per_thread(dtempl, cell):
    """The documented threading model: one Indexer each, GIL released."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(11)
    patterns = [synthetic_peaks(dtempl, cell, rng) for _ in range(4)]

    def work(peaks):
        indexer = crystfel.Indexer('xgandalf', dtempl, cell)
        img = crystfel.Image(dtempl)
        img.set_peaks(peaks[0], peaks[1], 0, 1.0)
        return indexer.index(img)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, patterns))

    assert all(n >= 1 for n in results)
