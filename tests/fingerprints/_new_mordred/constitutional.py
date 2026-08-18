import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.constitutional import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were generated with mordred-community and are stored at
./references/constitutional.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "constitutional.json") as f:
    _REFERENCE = json.load(f)

_MOLECULES = list(_REFERENCE["expected"])
_FEATURE_NAMES = _REFERENCE["feature_names"]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols):
    computed = {}
    for name in _MOLECULES:
        mol = preprocess_mol(mordred_test_mols[name], explicit_hydrogens=True)
        values = calc(mol)
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", _MOLECULES)
@pytest.mark.parametrize("feature_name", _FEATURE_NAMES)
def test_constitutional_reference_values(feature_name, molecule, computed_values):
    expected = _REFERENCE["expected"][molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, rtol=1e-6, equal_nan=True)


def test_feature_names():
    assert set(_FEATURE_NAMES).issubset(set(FEATURE_NAMES))
