import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles, SmilesParserParams

from skfp.fingerprints._new_mordred.descriptors import atom_count
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

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
    values = _calc(MolFromSmiles(smiles))
    assert_allclose(values, np.asarray(expected, dtype=np.float32))


def _parse_keeping_hydrogens(smiles: str) -> Mol:
    params = SmilesParserParams()
    params.removeHs = False
    return MolFromSmiles(smiles, params)


@pytest.mark.parametrize(
    ("mol", "expected_num_heavy_atoms", "expected_num_hydrogens"),
    [
        # hydrogens that RemoveHs keeps as atoms still count, as in Mordred: as
        # hydrogens, and as atoms of the hydrogen-suppressed molecule
        (MolFromSmiles("[2H]c1ccccc1C"), 8, 8),
        (_parse_keeping_hydrogens("[H]/N=C/C"), 4, 5),
        (_parse_keeping_hydrogens("[H][H]"), 2, 2),
        (MolFromSmiles("[H-].[Na+]"), 2, 1),
    ],
)
def test_hydrogens_kept_by_remove_hs_are_counted(
    mol, expected_num_heavy_atoms, expected_num_hydrogens
):
    values = _calc(mol)
    assert values[FEATURE_NAMES.index("nHeavyAtom")] == expected_num_heavy_atoms
    assert values[FEATURE_NAMES.index("nH")] == expected_num_hydrogens


def _calc(mol: Mol) -> np.ndarray:
    mol_regular = preprocess_mol(mol)
    return atom_count.calc(mol_regular, AtomicProperties(mol_regular))
