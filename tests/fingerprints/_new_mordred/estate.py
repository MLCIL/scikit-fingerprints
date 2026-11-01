import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.estate import FEATURE_NAMES, calc

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.fixture(scope="module")
def histidine_estate_features() -> dict[str, float]:
    # histidine tests in Mordred use a different tautomer from regular SDF file one
    mol = MolFromSmiles("NC(Cc1c[nH]cn1)C(=O)O")
    values = calc(mol)
    return dict(zip(FEATURE_NAMES, values, strict=True))


@pytest.mark.parametrize(
    "feature_name, expected_value",
    [
        ("number of dO atoms", 1),
        ("number of sOH atoms", 1),
        ("number of sNH2 atoms", 1),
        ("number of aaN atoms", 1),
        ("number of aaNH atoms", 1),
        ("number of aaCH atoms", 2),
        ("number of aasC atoms", 1),
        ("number of sssCH atoms", 1),
        ("sum of dO E-states", 10.263),
        ("sum of sOH E-states", 8.418),
        ("sum of sNH2 E-states", 5.257),
        ("sum of aaN E-states", 3.840),
        ("sum of aaNH E-states", 2.714),
        ("sum of aasC E-states", 0.666),
        ("sum of sssCH E-states", -0.863),
        ("sum of dssC E-states", -1.006),
        ("sum of sCH3 E-states", 0),
    ],
)
def test_estate_reference_values(
    feature_name, expected_value, histidine_estate_features
):
    assert_allclose(histidine_estate_features[feature_name], expected_value, atol=1e-2)
