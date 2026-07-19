import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags

from skfp.fingerprints._new_mordred.descriptors.distance_matrix import calc
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from PaDEL-Descriptor and stored at
./references/distance_matrix.json as {"digits": {descriptor: digit},
"values": {molecule: {descriptor: value}}}.

``digit`` follows the mordred-community reference-test convention, where it is
the number of decimal places compared via numpy.testing.assert_almost_equal
(threshold 1.5 * 10**-digit). Unlike mordred, which computes in float64, skfp
descriptors are float32, so large-magnitude spectral descriptors (e.g. SpAD_D
for Digoxin, ~1126) cannot meet a strict absolute digit=5 tolerance. We
therefore derive the tolerance from ``digit`` but apply it as both atol and
rtol, letting the rtol term absorb float32 storage error on large values.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "distance_matrix.json") as f:
    _REFERENCE = json.load(f)

_DIGITS = _REFERENCE["digits"]
_VALUES = _REFERENCE["values"]

_PARAMS = [
    (molecule, feature_name)
    for molecule, features in _VALUES.items()
    for feature_name in features
]


def _param(molecule, feature_name):
    # LogEE intentionally diverges from the PaDEL reference, whose LogEE (like
    # mordred-community's) adds a spurious exp(-a) term and computes
    # log(1 + sum(exp(lambda_i))) instead of the documented
    # log(sum(exp(lambda_i))). The divergence is log(1 + 1/EE), so large-EE
    # molecules still match within tolerance and xpass; hence strict=False.
    # See https://github.com/JacksonBurns/mordred-community/issues/24.
    if feature_name == "LogEE_D":
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
    for name in _VALUES:
        mol = mordred_test_mols[name]
        n_frags = len(GetMolFrags(mol))
        mol_regular = preprocess_mol(mol)
        distance_matrix_regular = DistanceMatrix(mol_regular)
        values, feature_names = calc(mol_regular, n_frags, distance_matrix_regular)
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize(
    "molecule, feature_name",
    [_param(molecule, feature_name) for molecule, feature_name in _PARAMS],
)
def test_distance_matrix_reference_values(feature_name, molecule, computed_values):
    expected = _VALUES[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    # `digit` is mordred's decimal-place precision; applied as both atol and
    # rtol so the rtol term absorbs float32 storage error on large descriptors
    # and the atol term absorbs reference rounding on small ones.
    tol = 10 ** -_DIGITS[feature_name]
    assert_allclose(actual, expected, rtol=tol, atol=tol, equal_nan=True)
