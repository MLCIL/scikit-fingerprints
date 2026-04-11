import pytest
from numpy.testing import assert_almost_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors.abc_index import (
    _calc_abc_index,
    _calc_abcgg_index,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

REFERENCE_DATA = [
    ("CC(C)CCCCCCC", 6.58, 6.49),
    ("CCC(C)CCCCCC", 6.47, 6.58),
    ("CC(C)(C)CCCCCC", 6.84, 6.82),
    ("CCC(C)(C)CCCCC", 6.68, 6.95),
]


@pytest.mark.parametrize(
    "smi,expected_abc,expected_abcgg",
    REFERENCE_DATA,
    ids=[row[0] for row in REFERENCE_DATA],
)
def test_abc_index_reference_values(smi, expected_abc, expected_abcgg):
    """
    Check ABC index against reference values from
    :doi:`10.2298/JSC150901093F`.
    """
    mol = Chem.MolFromSmiles(smi)
    dm = DistanceMatrix(mol)

    assert_almost_equal(_calc_abc_index(mol), expected_abc, decimal=2)
    assert_almost_equal(_calc_abcgg_index(mol, dm), expected_abcgg, decimal=2)
