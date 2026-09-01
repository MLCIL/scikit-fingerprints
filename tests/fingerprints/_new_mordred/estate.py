import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.estate import (
    FEATURE_NAMES,
    calc,
    calc_indices,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.fixture(scope="module")
def histidine_estate_features() -> dict[str, float]:
    # histidine tests in Mordred use a different tautomer from regular SDF file one
    mol = MolFromSmiles("NC(Cc1c[nH]cn1)C(=O)O")
    indices = calc_indices(AtomicProperties.from_mol(mol), DistanceMatrix.from_mol(mol))
    values = calc(mol, indices)
    return dict(zip(FEATURE_NAMES, values, strict=True))


@pytest.mark.parametrize(
    "feature_name, expected_value",
    [
        ("NdO", 1),
        ("NsOH", 1),
        ("NsNH2", 1),
        ("NaaN", 1),
        ("NaaNH", 1),
        ("NaaCH", 2),
        ("NaasC", 1),
        ("NsssCH", 1),
        ("SdO", 10.263),
        ("SsOH", 8.418),
        ("SsNH2", 5.257),
        ("SaaN", 3.840),
        ("SaaNH", 2.714),
        ("SaasC", 0.666),
        ("SsssCH", -0.863),
        ("SdssC", -1.006),
        ("SsCH3", 0),
    ],
)
def test_estate_reference_values(
    feature_name, expected_value, histidine_estate_features
):
    assert_allclose(histidine_estate_features[feature_name], expected_value, atol=1e-2)
