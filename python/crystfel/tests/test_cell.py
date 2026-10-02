import numpy as np
import pytest

import crystfel


def test_parameters_round_trip():
    cell = crystfel.UnitCell(63.0, 63.0, 42.0, 90, 90, 120,
                             lattice_type='hexagonal', unique_axis='c')
    a, b, c, al, be, ga = cell.parameters
    assert (a, b, c) == pytest.approx((63.0, 63.0, 42.0))
    assert (al, be, ga) == pytest.approx((90.0, 90.0, 120.0))
    assert cell.lattice_type == 'hexagonal'
    assert cell.centering == 'P'
    assert cell.unique_axis == 'c'


def test_volume_is_in_cubic_angstroms():
    cell = crystfel.UnitCell(63.0, 63.0, 42.0, 90, 90, 120,
                             lattice_type='hexagonal', unique_axis='c')
    expected = 63.0 * 63.0 * 42.0 * np.sin(np.deg2rad(120.0))
    assert cell.volume == pytest.approx(expected, rel=1e-9)


def test_resolution_is_half_of_one_over_d():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')
    assert cell.d_spacing(1, 0, 0) == pytest.approx(50.0)
    assert cell.resolution(1, 0, 0) == pytest.approx(1.0 / 100.0)
    assert cell.d_spacing(2, 0, 0) == pytest.approx(25.0)


def test_reciprocal_axes():
    cell = crystfel.UnitCell(50.0, 60.0, 70.0, 90, 90, 90,
                             lattice_type='orthorhombic')
    rec = cell.reciprocal_axes
    assert np.linalg.norm(rec[0]) == pytest.approx(1 / 50.0)
    assert np.linalg.norm(rec[1]) == pytest.approx(1 / 60.0)
    assert np.linalg.norm(rec[2]) == pytest.approx(1 / 70.0)


def test_direct_axes():
    cell = crystfel.UnitCell(50.0, 60.0, 70.0, 90, 90, 90,
                             lattice_type='orthorhombic')
    direct = cell.direct_axes
    assert np.linalg.norm(direct[0]) == pytest.approx(50.0)
    assert np.linalg.norm(direct[2]) == pytest.approx(70.0)


def test_validate_accepts_a_consistent_cell():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')
    cell.validate()
    assert cell.bravais_lattice
    assert cell.is_sensible
    assert cell.right_handed


def test_validate_rejects_a_non_bravais_combination():
    """Cubic A-centred is not one of the 14 Bravais lattices."""
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             centering='A', lattice_type='cubic')
    with pytest.raises(crystfel.CrystFELError, match='Bravais'):
        cell.validate()


def test_validate_rejects_centering_on_the_unique_axis():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             centering='C', lattice_type='monoclinic',
                             unique_axis='c')
    with pytest.raises(crystfel.CrystFELError):
        cell.validate()


def test_validate_does_not_check_the_parameters():
    """Worth pinning down, because it is surprising: validate_cell() looks at
    the lattice type, centering and unique axis, not at a, b and c."""
    cell = crystfel.UnitCell(50.0, 60.0, 70.0, 90, 90, 90,
                             lattice_type='cubic')
    cell.validate()


def test_unknown_lattice_type():
    with pytest.raises(ValueError):
        crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                          lattice_type='dodecahedral')


def test_file_round_trip(tmp_path):
    cell = crystfel.UnitCell(63.0, 63.0, 42.0, 90, 90, 120,
                             lattice_type='hexagonal', unique_axis='c')
    path = tmp_path / 'test.cell'
    cell.write(path)
    again = crystfel.UnitCell.from_file(path)
    assert again.parameters == pytest.approx(cell.parameters)
    assert again.lattice_type == cell.lattice_type
    assert again.unique_axis == cell.unique_axis


def test_missing_cell_file():
    with pytest.raises(crystfel.CrystFELError):
        crystfel.UnitCell.from_file('/nonexistent/test.cell')


def test_transform_doubles_an_axis():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')
    doubled = cell.transform([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert doubled.parameters[0] == pytest.approx(100.0)
    assert doubled.parameters[1] == pytest.approx(50.0)
    # The original is untouched
    assert cell.parameters[0] == pytest.approx(50.0)


def test_transform_rejects_non_integer():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')
    with pytest.raises(ValueError):
        cell.transform([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_uncenter_a_centred_cell():
    """An I-centred cubic cell has half the volume when made primitive."""
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             centering='I', lattice_type='cubic')
    primitive, matrix = cell.uncenter()
    assert primitive.volume == pytest.approx(cell.volume / 2.0, rel=1e-9)
    # CrystFEL describes the primitive cell of body-centred cubic in the
    # rhombohedral setting
    assert primitive.lattice_type == 'rhombohedral'
    assert matrix.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]


def test_matches_within_tolerance():
    a = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90, lattice_type='cubic')
    close = crystfel.UnitCell(50.5, 50.0, 50.0, 90, 90, 90,
                              lattice_type='cubic')
    far = crystfel.UnitCell(70.0, 50.0, 50.0, 90, 90, 90,
                            lattice_type='cubic')
    assert a.matches(close)
    assert not a.matches(far)


def test_matches_tolerance_angles_are_degrees():
    a = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90, lattice_type='cubic')
    tilted = crystfel.UnitCell(50.0, 50.0, 50.0, 91.0, 90, 90,
                               lattice_type='triclinic')
    assert a.matches(tilted, tolerances=(0.05, 0.05, 0.05, 2.0, 2.0, 2.0))
    assert not a.matches(tilted, tolerances=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5))


def test_impose_bravais():
    almost = crystfel.UnitCell(50.1, 49.9, 50.0, 90.1, 89.9, 90.0,
                               lattice_type='triclinic')
    cubic = almost.impose_bravais('cubic')
    a, b, c, al, be, ga = cubic.parameters
    assert a == pytest.approx(b)
    assert b == pytest.approx(c)
    assert (al, be, ga) == pytest.approx((90.0, 90.0, 90.0))


def test_copy_is_independent():
    cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                             lattice_type='cubic')
    other = cell.copy()
    other.centering = 'I'
    assert cell.centering == 'P'
    assert other.centering == 'I'


def test_reindexed_to_finds_an_alternative():
    """A tetragonal cell described with the unique axis permuted should be
    recognised as the same lattice after reindexing."""
    reference = crystfel.UnitCell(40.0, 40.0, 80.0, 90, 90, 90,
                                  lattice_type='tetragonal', unique_axis='c')
    permuted = crystfel.UnitCell(80.0, 40.0, 40.0, 90, 90, 90,
                                 lattice_type='tetragonal', unique_axis='a')
    assert not permuted.matches(reference)
    found, matrix = permuted.reindexed_to(reference)
    assert found is not None
    assert found.matches(reference)
    assert matrix.shape == (3, 3)


def test_lattice_types_constant():
    assert 'hexagonal' in crystfel.LATTICE_TYPES
    for name in crystfel.LATTICE_TYPES:
        cell = crystfel.UnitCell(50.0, 50.0, 50.0, 90, 90, 90,
                                 lattice_type=name)
        assert cell.lattice_type == name
