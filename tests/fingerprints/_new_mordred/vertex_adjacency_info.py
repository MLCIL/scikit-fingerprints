import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import vertex_adjacency_info
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = ["VAdjMat"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("C", np.nan),
        ("CC", 1.0),
        ("CCCC", 1 + np.log2(3)),
        ("CCO", 1 + np.log2(2)),
        ("c1ccccc1", 1 + np.log2(6)),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", 1 + np.log2(13)),
    ],
)
def test_vertex_adjacency_info_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)

    values, feature_names = vertex_adjacency_info.calc(AtomicProperties(mol_regular))

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, np.float32(expected), rtol=1e-6)
