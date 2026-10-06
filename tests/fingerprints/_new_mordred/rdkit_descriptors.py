import ast
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import MolSurf
from rdkit.Chem.EState import EState_VSA

from skfp.fingerprints._new_mordred.descriptors import estate, rdkit_descriptors
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

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
        mol_regular, distance_matrix, estate_indices, mol_properties
    ):
        assert all(atom.GetAtomicNum() != 1 for atom in mol_regular.GetAtoms())
        return np.zeros(len(rdkit_descriptors.FEATURE_NAMES_2D), dtype=np.float32)

    monkeypatch.setattr(
        calculator.rdkit_descriptors,
        "calc_rdkit_2d",
        calc_rdkit_2d_without_explicit_hydrogens,
    )

    calculator.compute(Chem.MolFromSmiles("CCO"), use_3D=False)


def test_moe_type_descriptors_match_rdkit(mordred_test_mols):
    for name, mol in mordred_test_mols.items():
        mol = preprocess_mol(mol)
        props, distance_matrix = (
            AtomicProperties.from_mol(mol),
            DistanceMatrix.from_mol(mol),
        )
        estate_indices = estate.calc_indices(props, distance_matrix)

        expected = [
            *[getattr(MolSurf, f"PEOE_VSA{i}")(mol) for i in range(1, 14)],
            *[getattr(MolSurf, f"SMR_VSA{i}")(mol) for i in range(1, 10)],
            *[getattr(MolSurf, f"SlogP_VSA{i}")(mol) for i in range(1, 12)],
            *[getattr(EState_VSA, f"EState_VSA{i}")(mol) for i in range(1, 11)],
            *[getattr(EState_VSA, f"VSA_EState{i}")(mol) for i in range(1, 10)],
        ]

        assert_allclose(
            rdkit_descriptors._calc_moe_type_descriptors(mol, estate_indices),
            expected,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"MOE-type descriptors differ for molecule {name}",
        )
