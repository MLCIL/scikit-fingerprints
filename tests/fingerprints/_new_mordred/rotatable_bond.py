import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import rotatable_bond
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = ["nRot", "RotRatio"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CC", [0, 0]),
        ("CCCC", [1, 1 / 3]),
        ("CCO", [0, 0]),
        ("c1ccccc1", [0, 0]),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", [2, 2 / 13]),
    ],
)
def test_rotatable_bond_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)

    values = rotatable_bond.calc(mol_regular)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
