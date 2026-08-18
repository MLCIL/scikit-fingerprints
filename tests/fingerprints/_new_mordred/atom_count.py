import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors import atom_count

FEATURE_NAMES = [
    "nAtom",
    "nHeavyAtom",
    "nSpiro",
    "nBridgehead",
    "nHetero",
    "nH",
    "nB",
    "nC",
    "nN",
    "nO",
    "nS",
    "nP",
    "nF",
    "nCl",
    "nBr",
    "nI",
    "nX",
]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CCO", [9, 3, 0, 0, 1, 6, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("c1ccccc1", [12, 6, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C(F)(Cl)(Br)I", [5, 5, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4]),
        ("O=P(O)(O)O", [8, 5, 0, 0, 5, 3, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0]),
        ("CS(=O)(=O)N", [10, 5, 0, 0, 4, 5, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0]),
        ("C[N+](C)(C)C", [17, 5, 0, 0, 1, 12, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_atom_count_values(smiles, expected):
    mol = MolFromSmiles(smiles)

    values = atom_count.calc(mol)
    assert_allclose(values, np.asarray(expected, dtype=np.float32))
