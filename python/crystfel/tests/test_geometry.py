import numpy as np
import pytest

import crystfel
from conftest import GROUPED_GEOM


def test_parse_and_query(dtempl):
    assert dtempl.wavelength == pytest.approx(1e-10)
    assert dtempl.camera_length == pytest.approx(0.1)
    assert dtempl.slab_extents == (2048, 2048)
    assert dtempl.panel_name(0) == 'p0'
    assert dtempl.panel_number('p0') == 0


def test_bad_geometry_raises():
    with pytest.raises(crystfel.GeometryError):
        crystfel.DataTemplate.from_string("this is not a geometry file\n")


def test_missing_file_raises():
    with pytest.raises(crystfel.GeometryError):
        crystfel.DataTemplate('/nonexistent/detector.geom')


def test_unknown_panel(dtempl):
    with pytest.raises(KeyError):
        dtempl.panel_number('nosuchpanel')


def test_round_trip(tmp_path, grouped):
    """Writing a geometry and reading it back must be a fixed point.

    This mirrors the geom_roundtrip test in the C test suite.
    """
    out = tmp_path / 'written.geom'
    grouped.write(out)
    again = crystfel.DataTemplate(out)
    assert again.to_string() == grouped.to_string()


def test_to_string_matches_write(tmp_path, dtempl):
    out = tmp_path / 'written.geom'
    dtempl.write(out)
    assert out.read_text() == dtempl.to_string()


def test_group_hierarchy(grouped):
    groups = {g.name: g for g in grouped.groups}
    assert 'all' in groups
    assert groups['all'].hierarchy_level == 0
    assert not groups['all'].leaf
    assert groups['lower'].hierarchy_level == 1
    assert groups['p0'].leaf
    assert groups['p0'].hierarchy_level == 2


def test_translate_group_moves_only_that_group(grouped):
    before = {p.name: p.corner_m for p in grouped.detector().panels}
    grouped.translate_group('lower', 0.0, 0.0, 1e-3)
    after = {p.name: p.corner_m for p in grouped.detector().panels}

    for name in ('p0', 'p1'):
        assert after[name][2] - before[name][2] == pytest.approx(1e-3)
    for name in ('p2', 'p3'):
        assert after[name][2] == pytest.approx(before[name][2])


def test_translate_group_units(grouped):
    pitch = grouped.detector()['p0'].pixel_pitch
    before = grouped.detector()['p0'].corner_m[0]
    grouped.translate_group('p0', 10.0, 0.0, 0.0, units='px')
    after = grouped.detector()['p0'].corner_m[0]
    assert after - before == pytest.approx(10.0 * pitch)


def test_rotate_group_is_in_degrees(grouped):
    """A 90 degree rotation about z sends x to y (right handed)."""
    before = grouped.detector()['p0'].fs_vector
    grouped.rotate_group('all', 90.0, 'z')
    after = grouped.detector()['p0'].fs_vector
    assert after == pytest.approx((-before[1], before[0], before[2]),
                                  abs=1e-9)


def test_rotate_group_rejects_bad_axis(grouped):
    with pytest.raises(ValueError):
        grouped.rotate_group('all', 1.0, 'w')


def test_translate_group_unknown_name(grouped):
    with pytest.raises(crystfel.GeometryError):
        grouped.translate_group('nosuchgroup', 1.0, 0.0, 0.0)


def test_transformations_survive_a_round_trip(tmp_path, grouped):
    grouped.translate_group('upper', 1e-4, -2e-4, 3e-4)
    grouped.rotate_group('lower', 0.25, 'z')
    out = tmp_path / 'moved.geom'
    grouped.write(out)

    # Geometry files store corners and scan vectors with "%f", so the
    # round trip is only exact to six decimal places of a pixel.
    reloaded = crystfel.DataTemplate(out)
    for a, b in zip(grouped.detector().panels, reloaded.detector().panels):
        assert a.corner_m == pytest.approx(b.corner_m, abs=1e-9)
        assert a.fs_vector == pytest.approx(b.fs_vector, abs=1e-6)


def test_coordinate_round_trip(grouped):
    """panel -> file -> panel must be the identity."""
    fs = np.array([0.5, 100.25, 511.5])
    ss = np.array([0.5, 7.75, 400.0])
    ffs, fss = grouped.panel_to_file_coords(3, fs, ss)
    back_fs, back_ss = grouped.file_to_panel_coords(ffs, fss, 3)
    assert back_fs == pytest.approx(fs, abs=1e-3)
    assert back_ss == pytest.approx(ss, abs=1e-3)


def test_coordinate_scalars(grouped):
    fs, ss = grouped.panel_to_file_coords(3, 10.5, 20.5)
    assert isinstance(fs, float) and isinstance(ss, float)
    # Panel 3 starts at (512, 512) in the slab
    assert fs == pytest.approx(522.5)
    assert ss == pytest.approx(532.5)


def test_slabby_finds_the_panel(grouped):
    fs, ss, pn = grouped.slabby_file_to_panel_coords(
        np.array([10.0, 600.0, 10.0, 600.0]),
        np.array([10.0, 10.0, 600.0, 600.0]))
    assert list(pn) == [0, 1, 2, 3]
    assert fs == pytest.approx([10.0, 88.0, 10.0, 88.0])


def test_coordinate_shapes_must_match(grouped):
    with pytest.raises(ValueError):
        grouped.file_to_panel_coords(np.zeros(3), np.zeros(4), 0)


def test_detector_access(grouped):
    det = grouped.detector()
    assert len(det) == 4
    assert [p.name for p in det.panels] == ['p0', 'p1', 'p2', 'p3']
    assert det['p2'].name == 'p2'
    assert det[-1].name == 'p3'
    with pytest.raises(KeyError):
        det['nope']
    with pytest.raises(IndexError):
        det[4]


def test_panel_properties(dtempl):
    panel = dtempl.detector()[0]
    assert panel.shape == (2048, 2048)
    assert panel.pixel_pitch == pytest.approx(1e-4)
    assert panel.adu_per_photon == pytest.approx(1.0)
    assert panel.fs_vector == pytest.approx((1.0, 0.0, 0.0))
    assert panel.ss_vector == pytest.approx((0.0, 1.0, 0.0))
    assert panel.corner_m[2] == pytest.approx(0.1)


def test_panel_group_name(grouped):
    assert grouped.detector()['p0'].group == 'p0'


def test_mean_camera_length(dtempl):
    assert dtempl.detector().mean_camera_length == pytest.approx(0.1)


def test_detector_needs_clen():
    """A geometry whose camera length comes from a header has no detector
    until a file is read, unless we say we do not care."""
    geom = GROUPED_GEOM.replace('clen = 100 mm', 'clen = /LCLS/clen')
    dt = crystfel.DataTemplate.from_string(geom)
    assert dt.camera_length is None
    assert dt.detector() is None
    assert dt.detector(no_clen_ok=True) is not None


def test_transform_coords_direct_beam(dtempl):
    """The direct beam position must map to the origin of reciprocal space."""
    det = dtempl.detector()
    q = det.transform_coords(0, 1024.0, 1024.0, dtempl.wavelength)
    assert q.shape == (3,)
    assert np.linalg.norm(q) < 1.0        # m^-1, i.e. essentially zero


def test_transform_coords_magnitude(dtempl):
    """|q| must equal 2 sin(theta) / lambda."""
    det = dtempl.detector()
    panel = det[0]
    wl = dtempl.wavelength
    fs = 1024.0 + 300.0
    q = det.transform_coords(panel, fs, 1024.0, wl)

    offset = 300.0 * panel.pixel_pitch
    two_theta = np.arctan2(offset, panel.corner_m[2])
    assert np.linalg.norm(q) == pytest.approx(2 * np.sin(two_theta / 2) / wl,
                                              rel=1e-9)


def test_transform_coords_vectorised(dtempl):
    det = dtempl.detector()
    fs = np.linspace(10, 2000, 17).reshape(17, 1)
    ss = np.full((17, 1), 1024.0)
    q = det.transform_coords(0, fs, ss, dtempl.wavelength)
    assert q.shape == (17, 1, 3)
    single = det.transform_coords(0, float(fs[5, 0]), 1024.0, dtempl.wavelength)
    assert q[5, 0] == pytest.approx(single)


def test_max_resolution(dtempl):
    det = dtempl.detector()
    corner = det.transform_coords(0, 0.0, 0.0, dtempl.wavelength)
    assert det.max_resolution(dtempl.wavelength) == pytest.approx(
        np.linalg.norm(corner), rel=1e-9)
