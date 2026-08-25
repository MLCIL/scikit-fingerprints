import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.wiener_index import (
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Hexane", [35, 3]),
        ("Benzene", [27, 3]),
        ("Caffeine", [258, 25]),
        ("Cyanidin", [894, 36]),
        ("Lycopene", [8904, 43]),
        ("Epicatechin", [894, 36]),
        ("Limonene", [120, 11]),
        ("Allicin", [108, 7]),
        ("Glutathione", [969, 24]),
        ("Digoxin", [14940, 113]),
        ("Capsaicin", [1411, 26]),
        ("EllagicAcid", [851, 48]),
        ("Astaxanthin", [10234, 69]),
    ],
)
def test_wiener_index_values(name, expected, mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols[name])
    distance_matrix_regular = DistanceMatrix.from_mol(mol_regular)

    values = calc(mol_regular, distance_matrix_regular)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
