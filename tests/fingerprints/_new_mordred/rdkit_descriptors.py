import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import rdkit_descriptors
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)

RDKIT_2D_FEATURE_NAMES = [
    "BalabanJ",
    "BertzCT",
    "nHBAcc",
    "nHBDon",
    "LabuteASA",
    *[f"PEOE VSA {i}" for i in range(1, 14)],
    *[f"SMR VSA {i}" for i in range(1, 10)],
    *[f"SlogP VSA {i}" for i in range(1, 12)],
    *[f"EState VSA {i}" for i in range(1, 11)],
    *[f"VSA EState {i}" for i in range(1, 10)],
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

RDKIT_3D_FEATURE_NAMES = [
    "principal moment of inertia Z",
    "principal moment of inertia Y",
    "principal moment of inertia X",
    "PBF",
]


def test_rdkit_descriptors_avoid_lambda_wrappers():
    source = Path(rdkit_descriptors.__file__).read_text()
    tree = ast.parse(source)

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]


def test_2d_calculator_passes_both_molecules_to_rdkit_descriptors(monkeypatch):
    from skfp.fingerprints._new_mordred import calculator

    def calc_rdkit_2d_checking_molecules(
        mol_regular, mol_hydrogens, distance_matrix, mol_properties
    ):
        assert all(atom.GetAtomicNum() != 1 for atom in mol_regular.GetAtoms())
        assert any(atom.GetAtomicNum() == 1 for atom in mol_hydrogens.GetAtoms())
        return np.zeros(len(rdkit_descriptors.FEATURE_NAMES_2D), dtype=np.float32)

    monkeypatch.setattr(
        calculator.rdkit_descriptors,
        "calc_rdkit_2d",
        calc_rdkit_2d_checking_molecules,
    )

    calculator.compute(Chem.MolFromSmiles("CCO"), use_3D=False)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # TopoPSA(NO) TopoPSA; Mordred reads the input as given, which is not
        # re-sanitized, so the contributions are those of non-aromatic atoms
        ("c1cnc[nH]1", [24.39, 24.39]),
        ("c1nn[nH]n1", [49.11, 49.11]),
        ("c1ccsc1", [0.0, 25.3]),
        ("Cc1ccncc1", [12.36, 12.36]),
    ],
)
def test_topological_polar_surface_area_of_kekulized_input(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol_regular = preprocess_mol(mol)

    values = rdkit_descriptors.calc_rdkit_2d(
        mol_regular,
        preprocess_mol(mol, explicit_hydrogens=True),
        DistanceMatrix.from_mol(mol_regular),
        MolecularProperties(mol_regular),
    )
    names = rdkit_descriptors.FEATURE_NAMES_2D
    tpsa = [values[names.index("TopoPSA(NO)")], values[names.index("TopoPSA")]]
    assert_allclose(tpsa, expected, rtol=1e-4)
