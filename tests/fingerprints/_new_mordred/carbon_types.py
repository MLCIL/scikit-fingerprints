import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import carbon_types
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = [
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
    "FCSP3",
]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("C#CC", [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1 / 3]),
        ("C=C", [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("CC=C(C)C", [0, 0, 0, 1, 1, 3, 0, 0, 0, 3 / 5, 3 / 5]),
        ("CC(C)(C)C", [0, 0, 0, 0, 0, 4, 0, 0, 1, 1, 1]),
        ("c1ccccc1", [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]),
        ("CCO", [0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1]),
    ],
)
def test_carbon_type_values(smiles, expected):
    values = _calc(Chem.MolFromSmiles(smiles))
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)


def _kekulized_without_aromatic_flags(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


@pytest.mark.parametrize(
    ("mol", "expected_fraction"),
    [
        # carbons with four neighbors, hydrogens included, not SP3-hybridized ones
        (Chem.MolFromSmiles("C[CH2-]"), 0.5),
        (Chem.MolFromSmiles("[CH2-][Tl]"), 0.0),
        (_kekulized_without_aromatic_flags("Cc1ccncc1"), 1 / 6),
        (Chem.MolFromSmiles("O"), 0.0),
    ],
)
def test_fraction_csp3_values(mol, expected_fraction):
    values = _calc(mol)
    assert_allclose(values[FEATURE_NAMES.index("FCSP3")], expected_fraction, rtol=1e-6)


def _calc(mol: Chem.Mol) -> np.ndarray:
    mol_kekulized = preprocess_mol(mol, explicit_hydrogens=False, kekulize=True)
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    return carbon_types.calc(mol_kekulized, mol_hydrogens)
