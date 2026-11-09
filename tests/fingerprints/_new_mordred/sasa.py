import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Atom, Conformer, Mol, RWMol
from rdkit.Geometry import Point3D

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func
from skfp.fingerprints._new_mordred.utils.periodic_table import VAN_DER_WAALS_RADII
from skfp.fingerprints._new_mordred.utils.sasa import solvent_accessible_surface_area

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _sasa(mol: Mol, conformer: int = -1) -> np.ndarray:
    """
    SASA with the atomic numbers and 3D distances built as the calculator does.
    """
    atomic_nums = atoms_apply_func(Atom.GetAtomicNum, mol, np.intp)
    distances = DistanceMatrix3D(mol, conformer).matrix
    return solvent_accessible_surface_area(mol, atomic_nums, distances, conformer)


@pytest.mark.parametrize(
    "name, expected_value",
    [
        ("Hexane", 296.910),
        ("Benzene", 243.552),
        ("Caffeine", 369.973),
        ("Cyanidin", 483.873),
        ("Lycopene", 1172.253),
        ("Epicatechin", 489.498),
        ("Limonene", 361.278),
        ("Allicin", 356.872),
        ("Glutathione", 530.679),
        ("Digoxin", 1074.428),
        ("Capsaicin", 641.527),
        ("EllagicAcid", 440.267),
        ("Astaxanthin", 1080.941),
        ("DMSO", 227.926),
        ("DiethylThioketone", 290.503),
        ("VinylsulfonicAcid", 246.033),
        ("Thiophene", 227.046),
        ("Triethoxyphosphine", 396.482),
        ("MethylphosphonicAcid", 235.685),
        ("MethylCyclopropane", 229.071),
        ("Acetonitrile", 182.197),
        ("Histidine", 335.672),
    ],
)
def test_sasa_reference_values(name, expected_value, mordred_test_mols_hydrogens_3d):
    mol = mordred_test_mols_hydrogens_3d[name]
    actual_value = _sasa(mol).sum()
    # just above the 5% of mordred's own SASA test: FreeSASA's Shrake-Rupley tests
    # fewer points per atom than Mordred's own and reads totals up to ~6% low
    assert_allclose(actual_value, expected_value, rtol=0.06)


def _mol_from_positions(atomic_num: int, positions: list[tuple[float, ...]]):
    """Molecule of unbonded atoms placed exactly where given, with no sanitization."""
    mol = RWMol()
    for _ in positions:
        mol.AddAtom(Atom(atomic_num))
    conformer = Conformer(len(positions))
    for idx, position in enumerate(positions):
        conformer.SetAtomPosition(idx, Point3D(*position))
    conformer.Set3D(True)

    out = mol.GetMol()
    out.AddConformer(conformer)
    return out


def _lone_sphere_area(atomic_num: int) -> float:
    """Area of a sphere no other atom reaches, the exact answer for a lone atom."""
    radii = VAN_DER_WAALS_RADII.lookup(np.array([atomic_num], dtype=np.intp))
    return 4.0 * math.pi * (float(radii[0]) + 1.4) ** 2


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([(0.0, 0.0, 0.0)], id="single_atom"),
        pytest.param([(0.0, 0.0, 0.0), (50.0, 0.0, 0.0)], id="two_far_apart"),
    ],
)
def test_sasa_lone_atoms_are_whole_spheres(positions):
    # RDKit's CalcSASA reads out of bounds on an atom that no other atom comes
    # within reach of, segfaulting the process or returning a wrong area, so such
    # atoms are kept away from it and get their exact analytical area instead
    mol = _mol_from_positions(6, positions)
    areas = _sasa(mol, conformer=0)

    assert_allclose(areas, _lone_sphere_area(6), rtol=1e-6)


def test_sasa_lone_atom_does_not_disturb_the_rest():
    # two carbons in contact plus one far away: the lone one is whole, and leaving
    # it out of the search does not change what the bonded pair comes out as
    far = 50.0
    pair = _mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    with_lone = _mol_from_positions(
        6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (far, 0.0, 0.0)]
    )

    areas = _sasa(with_lone, conformer=0)

    assert_allclose(areas[:2], _sasa(pair, conformer=0))
    assert_allclose(areas[2], _lone_sphere_area(6), rtol=1e-6)


def test_sasa_of_an_empty_molecule_is_empty():
    mol = _mol_from_positions(6, [])
    assert _sasa(mol, conformer=0).shape == (0,)


@pytest.mark.parametrize("atomic_num", [0, 104], ids=["dummy_atom", "rutherfordium"])
def test_sasa_is_nan_for_an_atom_without_radius(atomic_num):
    # Mordred's radii table has no entry for a dummy atom or for elements from
    # rutherfordium on; as in Mordred, such an atom has no area and covers no other,
    # so the carbon it would overlap stays a whole sphere
    mol = RWMol(_mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]))
    mol.GetAtomWithIdx(1).SetAtomicNum(atomic_num)

    areas = _sasa(mol, conformer=0)

    assert_allclose(areas[0], _lone_sphere_area(6), rtol=1e-6)
    assert np.isnan(areas[1])


def test_sasa_is_nan_when_no_atom_has_a_radius():
    mol = RWMol(_mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]))
    for atom in mol.GetAtoms():
        atom.SetAtomicNum(104)

    areas = _sasa(mol, conformer=0)

    assert np.isnan(areas).all()


def test_sasa_of_a_smaller_atom_inside_a_larger_one_is_zero():
    # the oxygen lies wholly inside the carbon, which keeps its whole sphere
    mol = RWMol(_mol_from_positions(6, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]))
    mol.GetAtomWithIdx(1).SetAtomicNum(8)

    areas = _sasa(mol, conformer=0)

    assert_allclose(areas, [_lone_sphere_area(6), 0.0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], id="stacked_pair"),
        pytest.param(
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
            id="stacked_pair_in_chain",
        ),
    ],
)
def test_sasa_of_stacked_atoms_is_finite(positions):
    # two atoms of the same radius at the same position, as when fragments are
    # embedded on top of each other, have each test point exactly on the other's
    # surface; like Mordred, which uses the same algorithm, the areas come down to
    # rounding, but they are finite and no larger than a whole sphere
    mol = _mol_from_positions(6, positions)

    areas = _sasa(mol, conformer=0)

    assert np.isfinite(areas).all()
    assert (areas >= 0.0).all()
    assert (areas <= _lone_sphere_area(6) * (1 + 1e-6)).all()


def test_sasa_reads_overlaps_from_the_given_distances():
    # two carbons in contact, but distances placing them far apart: the distances
    # decide which atoms are lone, so both come out as whole spheres
    mol = _mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    far_apart = np.array([[0.0, 50.0], [50.0, 0.0]])

    areas = solvent_accessible_surface_area(
        mol, np.array([6, 6]), far_apart, conformer=0
    )

    assert_allclose(areas, _lone_sphere_area(6), rtol=1e-6)


def test_sasa_reads_radii_from_the_given_atomic_numbers():
    # two carbons in contact, but atomic numbers making the second one an element
    # without radius: it gets no area, and leaves the first one a whole sphere
    mol = _mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    distances = DistanceMatrix3D(mol, 0).matrix

    areas = solvent_accessible_surface_area(
        mol, np.array([6, 104]), distances, conformer=0
    )

    assert_allclose(areas[0], _lone_sphere_area(6), rtol=1e-6)
    assert np.isnan(areas[1])
