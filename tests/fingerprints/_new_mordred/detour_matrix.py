import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags

from skfp.fingerprints._new_mordred.descriptors.detour_matrix import calc
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from PaDEL-Descriptor and stored at
./references/detour_matrix.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "detour_matrix.json") as f:
    _REFERENCE = json.load(f)

_PARAMS = [
    (molecule, feature_name)
    for molecule, features in _REFERENCE.items()
    for feature_name in features
]


def _param(molecule, feature_name):
    # LogEE intentionally diverges from the PaDEL reference, whose LogEE (like
    # mordred-community's) adds a spurious exp(-a) term and computes
    # log(1 + sum(exp(lambda_i))) instead of the documented
    # log(sum(exp(lambda_i))). The divergence is log(1 + 1/EE), so large-EE
    # molecules still match within tolerance and xpass; hence strict=False.
    # See https://github.com/JacksonBurns/mordred-community/issues/24.
    if feature_name == "LogEE_Dt":
        return pytest.param(
            molecule,
            feature_name,
            marks=pytest.mark.xfail(
                reason="LogEE fixed in skfp, PaDEL reference is buggy "
                "(https://github.com/JacksonBurns/mordred-community/issues/24)",
                strict=False,
            ),
        )
    return pytest.param(molecule, feature_name)


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols):
    computed = {}
    for name in _REFERENCE:
        mol = mordred_test_mols[name]
        n_frags = len(GetMolFrags(mol))
        mol_regular = preprocess_mol(mol)
        values, feature_names = calc(mol_regular, n_frags)
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize(
    "molecule, feature_name",
    [_param(molecule, feature_name) for molecule, feature_name in _PARAMS],
)
def test_detour_matrix_reference_values(feature_name, molecule, computed_values):
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, atol=1e-5, equal_nan=True)
