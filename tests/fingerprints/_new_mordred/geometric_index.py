import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.geometric_index import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected_values",
    [
        ("Hexane", [6.541237774, 3.885272976, 0.68359799, 0.406033979]),
        ("Benzene", [4.963197199, 3.876368801, 0.280372806, 0.218977476]),
        ("EllagicAcid", [10.346072306, 5.914824414, 0.749176574, 0.428302428]),
    ],
)
def test_geometric_index_reference_values(
    name, expected_values, mordred_test_mols_hydrogens_3d
):
    mol = mordred_test_mols_hydrogens_3d[name]
    dists = DistanceMatrix3D(mol)

    values, feature_names = calc(dists)
    values = dict(zip(feature_names, values, strict=True))
    values = [values[name] for name in FEATURE_NAMES]

    assert_allclose(values, expected_values, atol=1e-2)
