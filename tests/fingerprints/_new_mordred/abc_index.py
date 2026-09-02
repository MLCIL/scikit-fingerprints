import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.abc_index import (
    _calc_abc_index,
    _calc_abcgg_index,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    # values from 10.2298/JSC150901093F
    "smiles, expected_abc, expected_abcgg",
    [
        ("CC(C)CCCCCCC", 6.58, 6.49),
        ("CCC(C)CCCCCC", 6.47, 6.58),
        ("CC(C)(C)CCCCCC", 6.84, 6.82),
        ("CCC(C)(C)CCCCC", 6.68, 6.95),
    ],
)
def test_abc_index_reference_values(smiles, expected_abc, expected_abcgg):
    mol = MolFromSmiles(smiles)
    distance_matrix = DistanceMatrix.from_mol(mol)

    props = AtomicProperties.from_mol(mol)
    assert_allclose(_calc_abc_index(props), expected_abc, atol=1e-2)
    assert_allclose(
        _calc_abcgg_index(props, distance_matrix), expected_abcgg, atol=1e-2
    )
