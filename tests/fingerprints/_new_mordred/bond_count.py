import numpy as np
import pytest
from rdkit.Chem import Kekulize, Mol, MolFromSmiles, SmilesParserParams
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


def _calc(mol: Mol) -> np.ndarray:
    # inputs are built the same way calculator.compute builds them
    mol_regular = preprocess_mol(mol)
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    kekulized_bond_types, kekulized_bond_types_hydrogens = (
        bonds_apply_func(
            Bond.GetBondType,
            preprocess_mol(variant, kekulize=True, sanitize=False),
            np.intp,
        )
        for variant in (mol_regular, mol_hydrogens)
    )
    return calc(
        AtomicProperties(mol_regular),
        AtomicProperties(mol_hydrogens),
        kekulized_bond_types,
        kekulized_bond_types_hydrogens,
    )


def _kekulized_without_aromatic_flags(smiles: str) -> Mol:
    mol = MolFromSmiles(smiles)
    Kekulize(mol, clearAromaticFlags=True)
    return mol


def _parse_keeping_hydrogens(smiles: str) -> Mol:
    params = SmilesParserParams()
    params.removeHs = False
    return MolFromSmiles(smiles, params)


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
    values = _calc(MolFromSmiles(smiles))
    assert list(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mol", "expected"),
    [
        # nBonds nBondsO nBondsS nBondsD nBondsT nBondsA nBondsM nBondsKS nBondsKD
        # only the hydrogen-suppressed molecule is re-sanitized, so a Kekule input
        # has aromatic bonds there and single and double ones with hydrogens
        (
            _kekulized_without_aromatic_flags("c1ccc2ccccc2c1"),
            [19, 11, 14, 0, 0, 11, 11, 14, 5],
        ),
        (
            _kekulized_without_aromatic_flags("c1cnc[nH]1"),
            [9, 5, 7, 0, 0, 5, 5, 7, 2],
        ),
        # hydrogens that RemoveHs keeps as atoms bond to heavy atoms
        (MolFromSmiles("[2H]c1ccccc1C"), [15, 8, 9, 0, 0, 6, 6, 12, 3]),
        (_parse_keeping_hydrogens("[H]/N=C/C"), [7, 3, 6, 1, 0, 0, 1, 6, 1]),
        (_parse_keeping_hydrogens("[H][H]"), [1, 1, 1, 0, 0, 0, 0, 1, 0]),
    ],
)
def test_bond_count_values_of_inputs_unlike_remove_hs(mol, expected):
    values = _calc(mol)
    assert list(values) == pytest.approx(expected)


def test_bond_count_feature_names():
    values = _calc(MolFromSmiles("CCO"))
    assert len(values) == len(FEATURE_NAMES)
