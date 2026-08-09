import numpy as np
import pytest
from rdkit.Chem import AddHs, Kekulize, MolFromSmiles, RWMol

from skfp.fingerprints._new_mordred.descriptors.bond_count import FEATURE_NAMES, calc

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values computed using mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _calc(smiles: str) -> np.ndarray:
    mol = AddHs(MolFromSmiles(smiles))
    mol_kek = RWMol(AddHs(MolFromSmiles(smiles)))
    Kekulize(mol_kek)
    values = calc(mol, mol_kek)
    return values


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # nBonds nBondsO nBondsS nBondsD nBondsT nBondsA nBondsM nBondsKS nBondsKD
        ("CCO", [8, 2, 8, 0, 0, 0, 0, 8, 0]),
        ("c1ccccc1", [12, 6, 6, 0, 0, 6, 6, 9, 3]),
        ("C=C", [5, 1, 4, 1, 0, 0, 1, 4, 1]),
        ("C#N", [2, 1, 1, 0, 1, 0, 1, 1, 0]),
        ("CC(=O)O", [7, 3, 6, 1, 0, 0, 1, 6, 1]),
        ("c1ccc(N)cc1", [14, 7, 8, 0, 0, 6, 6, 11, 3]),
    ],
)
def test_bond_count_values(smiles, expected):
    values = _calc(smiles)
    assert list(values) == pytest.approx(expected)


def test_bond_count_feature_names():
    values = _calc("CCO")
    assert len(values) == len(FEATURE_NAMES)
