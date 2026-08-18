import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.eccentric_connectivity_index import (
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Hexane", 38),
        ("Benzene", 36),
        ("Caffeine", 137),
        ("Cyanidin", 353),
        ("Lycopene", 1834),
        ("Epicatechin", 353),
        ("Limonene", 88),
        ("Allicin", 83),
        ("Glutathione", 347),
        ("Digoxin", 2628),
        ("Capsaicin", 515),
        ("EllagicAcid", 328),
        ("Astaxanthin", 1932),
    ],
)
def test_eccentric_connectivity_index_values(name, expected, mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols[name])

    values = calc(AdjacencyMatrix(mol_regular), DistanceMatrix(mol_regular))
    assert_allclose(values, np.float32(expected), rtol=1e-6)
