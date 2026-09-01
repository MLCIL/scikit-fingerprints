import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import AddHs
from rdkit.Chem.rdchem import Bond

from skfp.fingerprints._new_mordred.descriptors.information_content import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    bonds_apply_func,
    preprocess_mol,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values are stored at ./references/information_content.json as
expected[molecule][descriptor], and cover only the orders and molecules Mordred's own
test suite does not skip. They are rounded to three decimals.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "information_content.json") as f:
    _REFERENCE = json.load(f)

_MOLECULES = list(_REFERENCE["expected"])
_FEATURE_NAMES = _REFERENCE["feature_names"]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols):
    computed = {}
    for name in _MOLECULES:
        mol_regular = preprocess_mol(mordred_test_mols[name])
        mol_hydrogens = AddHs(mol_regular)
        mol_kekulized = preprocess_mol(mordred_test_mols[name], kekulize=True)
        kekulized_bond_types = bonds_apply_func(
            Bond.GetBondType, mol_kekulized, np.intp
        )

        values = calc(
            mol_hydrogens,
            AtomicProperties.from_mol(mol_hydrogens),
            kekulized_bond_types,
        )
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", _MOLECULES)
@pytest.mark.parametrize("feature_name", _FEATURE_NAMES)
def test_information_content_reference_values(feature_name, molecule, computed_values):
    if feature_name not in _REFERENCE["expected"][molecule]:
        pytest.skip("mordred skips this order for this molecule")

    expected = _REFERENCE["expected"][molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    atol = 0.05 if feature_name.startswith("modified_information_content") else 1e-3
    assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)


def test_feature_names():
    assert set(_FEATURE_NAMES).issubset(set(FEATURE_NAMES))
