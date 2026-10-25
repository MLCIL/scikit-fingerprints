import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Atom, Conformer, RWMol
from rdkit.Geometry import Point3D

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    VAN_DER_WAALS_RADII_RDKIT,
)
from skfp.fingerprints._new_mordred.utils.sasa import solvent_accessible_surface_area

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


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
    actual_value = solvent_accessible_surface_area(mol).sum()
    # 5% relative tolerance, matching mordred's own SASA test
    assert_allclose(actual_value, expected_value, rtol=0.05)


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
    radii = VAN_DER_WAALS_RADII_RDKIT.lookup(np.array([atomic_num], dtype=np.intp))
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
    areas = solvent_accessible_surface_area(mol, conformer=0)

    assert_allclose(areas, _lone_sphere_area(6), rtol=1e-6)


def test_sasa_lone_atom_does_not_disturb_the_rest():
    # two carbons in contact plus one far away: the lone one is whole, and leaving
    # it out of the search does not change what the bonded pair comes out as
    far = 50.0
    pair = _mol_from_positions(6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    with_lone = _mol_from_positions(
        6, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (far, 0.0, 0.0)]
    )

    areas = solvent_accessible_surface_area(with_lone, conformer=0)

    assert_allclose(areas[:2], solvent_accessible_surface_area(pair, conformer=0))
    assert_allclose(areas[2], _lone_sphere_area(6), rtol=1e-6)
