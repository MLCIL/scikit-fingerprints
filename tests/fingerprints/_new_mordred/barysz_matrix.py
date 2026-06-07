import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.barysz_matrix import calc
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from mordred-community and stored at
./references/barysz_matrix.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_SMILES = {
    "Benzene": "c1ccccc1",
    "Hexane": "CCCCCC",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Limonene": "CC1=CCC(=CC1)C(C)=C",
}

with open(Path(__file__).parent / "references" / "barysz_matrix.json") as f:
    _REFERENCE = json.load(f)

_PROPERTIES = ["Z", "m", "v", "se", "pe", "are", "p", "i"]
_ATTRIBUTES = [
    "SpAbs",
    "SpMax",
    "SpDiam",
    "SpAD",
    "SpMAD",
    "LogEE",
    "SM1",
    "VE1",
    "VE2",
    "VE3",
    "VR1",
    "VR2",
    "VR3",
]


@pytest.fixture(scope="module")
def computed_values():
    computed = {}
    for name, smiles in _SMILES.items():
        mol = preprocess_mol(MolFromSmiles(smiles))
        values, feature_names = calc(mol, n_frags=1)
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", list(_SMILES))
@pytest.mark.parametrize("prop", _PROPERTIES)
@pytest.mark.parametrize("attr", _ATTRIBUTES)
def test_barysz_matrix_reference_values(attr, prop, molecule, computed_values):
    feature_name = f"{attr}_Dz{prop}"
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, atol=1e-3, equal_nan=True)
