import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Bond

from skfp.fingerprints._new_mordred.descriptors.bond_count import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    bonds_apply_func,
    preprocess_mol,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values computed using mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _calc(smiles: str) -> np.ndarray:
    mol_hydrogens = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)
    mol_kekulized_hydrogens = preprocess_mol(
        mol_hydrogens, kekulize=True, sanitize=False
    )
    kekulized_bond_types_hydrogens = bonds_apply_func(
        Bond.GetBondType, mol_kekulized_hydrogens, np.intp
    )
    values = calc(AtomicProperties(mol_hydrogens), kekulized_bond_types_hydrogens)
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
