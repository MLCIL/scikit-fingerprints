import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Kekulize, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.acid_base import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values computed using mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _kekulized_without_aromatic_flags(smiles: str) -> Mol:
    mol = MolFromSmiles(smiles)
    Kekulize(mol, clearAromaticFlags=True)
    return mol


def _calc(mol: Mol) -> np.ndarray:
    return calc(preprocess_mol(mol, explicit_hydrogens=True))


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # nAcid nBase
        ("CC(=O)O", [1, 0]),
        ("CCN", [0, 1]),
        ("CN(C)C", [0, 1]),
        ("NC(=N)N", [0, 3]),
        ("OS(=O)(=O)c1ccccc1", [1, 0]),
        ("C[NH3+].[Cl-]", [1, 1]),
        ("c1nn[nH]n1", [2, 0]),
        ("C[CH2-]", [1, 0]),
    ],
)
def test_acid_base_values(smiles, expected):
    assert_allclose(_calc(MolFromSmiles(smiles)), expected)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # the patterns see the Kekule bonds of the input, which is not re-sanitized
        ("c1cnc[nH]1", [0, 2]),
        ("c1nn[nH]n1", [0, 2]),
    ],
)
def test_acid_base_values_of_kekulized_input(smiles, expected):
    assert_allclose(_calc(_kekulized_without_aromatic_flags(smiles)), expected)


def test_acid_base_feature_names():
    assert len(_calc(MolFromSmiles("CCO"))) == len(FEATURE_NAMES)
