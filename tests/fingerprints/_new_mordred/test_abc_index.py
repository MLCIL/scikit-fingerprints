"""ABC index reference values from doi:10.2298/JSC150901093F."""

from numpy.testing import assert_almost_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors.abc_index import (
    _calc_abc_index,
    _calc_abcgg_index,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

REFERENCE_DATA = [
    ("CC(C)CCCCCCC", 6.58, 6.49),
    ("CCC(C)CCCCCC", 6.47, 6.58),
    ("CC(C)(C)CCCCCC", 6.84, 6.82),
    ("CCC(C)(C)CCCCC", 6.68, 6.95),
]


def test_abc_index_reference_values():
    for smi, expected_abc, expected_abcgg in REFERENCE_DATA:
        mol = Chem.MolFromSmiles(smi)
        dm = DistanceMatrix(mol)

        assert_almost_equal(_calc_abc_index(mol), expected_abc, decimal=2)
        assert_almost_equal(_calc_abcgg_index(mol, dm), expected_abcgg, decimal=2)
