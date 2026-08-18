import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.morse import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected_values",
    [
        ("Hexane", [190, 24.866, -4.391, 1.246, -2.878, -1.499, 0.357, 1.506]),
        ("Cyanidin", [496, 28.358, -2.358, 1.491, -4.704, -2.887, 4.308, -0.797]),
        ("EllagicAcid", [378, 21.726, -2.29, 1.323, -4.43, -1.414, 3.075, -1.444]),
    ],
)
def test_morse_unweighted_reference_values(
    name, expected_values, mordred_test_mols_hydrogens_3d
):
    mol = mordred_test_mols_hydrogens_3d[name]
    dists = DistanceMatrix3D(mol)

    values = calc(mol, dists)
    values = dict(zip(FEATURE_NAMES, values, strict=True))
    values = [
        values[f"MoRSE_unweighted_dist_{dist}"]
        for dist in range(1, len(expected_values) + 1)
    ]

    assert_allclose(values, expected_values, atol=1e-2)
