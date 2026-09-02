import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.cpsa import (
    FEATURE_NAMES_2D,
    FEATURE_NAMES_3D,
    calc_2d,
    calc_3d,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from mordred-community and stored at
./references/cpsa.json as expected[molecule][descriptor].

Surface-area-dependent descriptors are compared with rtol=0.05, matching the
5% relative tolerance mordred itself uses for SASA (see mordred/tests/test_SASA.py).
We share mordred's van der Waals radii, but integrate each atom's exposed fraction
on FreeSASA's 100 test points rather than mordred's 5112-point icosphere, which
leaves a few percent of quadrature noise per atom.

RNCS/RPCS get 20%: they divide by a single atom's surface area instead of the
total, so that per-atom noise reaches them undamped.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "cpsa.json") as f:
    _REFERENCE = json.load(f)

_MOLECULES = list(_REFERENCE)

_PER_ATOM_SASA = {"RNCS", "RPCS"}

_FEATURE_NAMES = [*FEATURE_NAMES_2D, *FEATURE_NAMES_3D]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols_hydrogens_3d):
    computed = {}
    for name in _MOLECULES:
        mol = mordred_test_mols_hydrogens_3d[name]
        charges = AtomicProperties.from_mol(mol).gasteiger_charges
        cpsa_2d = calc_2d(charges)
        values_3d = calc_3d(mol, cpsa_2d, charges)
        computed[name] = dict(zip(FEATURE_NAMES_2D, cpsa_2d, strict=True)) | dict(
            zip(FEATURE_NAMES_3D, values_3d, strict=True)
        )
    return computed


@pytest.mark.parametrize("molecule", _MOLECULES)
@pytest.mark.parametrize("feature_name", _FEATURE_NAMES)
def test_cpsa_reference_values(feature_name, molecule, computed_values):
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    rtol = 0.2 if feature_name in _PER_ATOM_SASA else 0.05
    assert_allclose(actual, expected, rtol=rtol, equal_nan=True)
