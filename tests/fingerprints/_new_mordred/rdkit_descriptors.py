import ast
from pathlib import Path

import numpy as np
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import rdkit_descriptors

RDKIT_2D_FEATURE_NAMES = [
    "BalabanJ",
    "BertzCT",
    "nHBAcc",
    "nHBDon",
    "LabuteASA",
    *[f"PEOE_VSA{i}" for i in range(1, 14)],
    *[f"SMR_VSA{i}" for i in range(1, 10)],
    *[f"SlogP_VSA{i}" for i in range(1, 12)],
    *[f"EState_VSA{i}" for i in range(1, 11)],
    *[f"VSA_EState{i}" for i in range(1, 10)],
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

RDKIT_3D_FEATURE_NAMES = ["MOMI-Z", "MOMI-Y", "MOMI-X", "PBF"]


def test_rdkit_descriptors_avoid_lambda_wrappers():
    source = Path(rdkit_descriptors.__file__).read_text()
    tree = ast.parse(source)

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]


def test_2d_calculator_passes_regular_molecule_to_rdkit_descriptors(monkeypatch):
    from skfp.fingerprints._new_mordred import calculator

    def calc_rdkit_2d_without_explicit_hydrogens(
        mol_regular, distance_matrix, mol_properties
    ):
        assert all(atom.GetAtomicNum() != 1 for atom in mol_regular.GetAtoms())
        return np.zeros(len(rdkit_descriptors.FEATURE_NAMES_2D), dtype=np.float32)

    monkeypatch.setattr(
        calculator.rdkit_descriptors,
        "calc_rdkit_2d",
        calc_rdkit_2d_without_explicit_hydrogens,
    )

    calculator.compute(Chem.MolFromSmiles("CCO"), use_3D=False)
