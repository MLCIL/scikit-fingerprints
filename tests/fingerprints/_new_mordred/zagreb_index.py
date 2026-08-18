import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.zagreb_index import (
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected_zagreb1",
    [
        ("Hexane", 18),
        ("Benzene", 24),
        ("Caffeine", 76),
        ("Cyanidin", 114),
        ("Lycopene", 170),
        ("Epicatechin", 114),
        ("Limonene", 46),
        ("Allicin", 32),
        ("Glutathione", 86),
        ("Digoxin", 320),
        ("Capsaicin", 98),
        ("EllagicAcid", 130),
        ("Astaxanthin", 218),
    ],
)
def test_zagreb1_values(name, expected_zagreb1, mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols[name])
    adjacency_matrix_regular = AdjacencyMatrix(mol_regular)

    values = calc(mol_regular, adjacency_matrix_regular)
    assert_allclose(values[0], np.float32(expected_zagreb1), rtol=1e-6)


def test_zagreb_all_features(mordred_test_mols):
    # full reference vector [Zagreb1, Zagreb2, mZagreb1, mZagreb2]
    expected = [18, 19, 1.61, 0.92]

    mol_regular = preprocess_mol(mordred_test_mols["MethylCyclopropane"])
    adjacency_matrix_regular = AdjacencyMatrix(mol_regular)

    values = calc(mol_regular, adjacency_matrix_regular)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), atol=1e-2)
