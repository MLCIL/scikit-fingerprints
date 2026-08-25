import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.barysz_matrix import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
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


def _feature_param(feature_name):
    # LogEE intentionally diverges from mordred-community, whose implementation
    # adds a spurious exp(-a) term and computes log(1 + sum(exp(lambda_i)))
    # instead of the documented log(sum(exp(lambda_i))).
    # See https://github.com/JacksonBurns/mordred-community/issues/24.
    if feature_name.startswith("LogEE"):
        return pytest.param(
            feature_name,
            marks=pytest.mark.xfail(
                reason="LogEE fixed in skfp, mordred-community reference is buggy "
                "(https://github.com/JacksonBurns/mordred-community/issues/24). "
                "Divergence is log(1 + 1/EE), so large-EE cases still match within "
                "atol and xpass; not strict.",
                strict=False,
            ),
        )
    return feature_name


@pytest.fixture(scope="module")
def computed_values():
    computed = {}
    for name, smiles in _SMILES.items():
        mol = preprocess_mol(MolFromSmiles(smiles))
        values = calc(mol, AtomicProperties.from_mol(mol), n_frags=1)
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", list(_SMILES))
@pytest.mark.parametrize(
    "feature_name", [_feature_param(name) for name in FEATURE_NAMES]
)
def test_barysz_matrix_reference_values(feature_name, molecule, computed_values):
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, atol=1e-3, equal_nan=True)
