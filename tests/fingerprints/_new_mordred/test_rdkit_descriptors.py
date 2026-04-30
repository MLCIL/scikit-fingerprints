import ast
from pathlib import Path

import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import rdkit_descriptors
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)
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

RDKIT_3D_FEATURE_NAMES = ["MOMI-X", "MOMI-Y", "MOMI-Z", "PBF"]


def test_rdkit_descriptors_avoid_lambda_wrappers():
    source = Path(rdkit_descriptors.__file__).read_text()
    tree = ast.parse(source)

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]


def test_2d_calculator_does_not_add_explicit_hydrogens(monkeypatch):
    from skfp.fingerprints._new_mordred import cache, calculator

    original_preprocess_mol = cache.preprocess_mol

    def preprocess_without_explicit_hydrogens(*args, **kwargs):
        assert kwargs.get("explicit_hydrogens", False) is False
        return original_preprocess_mol(*args, **kwargs)

    monkeypatch.setattr(cache, "preprocess_mol", preprocess_without_explicit_hydrogens)

    calculator.compute(Chem.MolFromSmiles("CCO"), use_3D=False)


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCC",
        "c1ccccc1",
        "CC=O",
        "CCN(CC)CC",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "O=S(=O)(O)O",
        "O",
        "[NH4+]",
        "[Na+].[Cl-]",
        "C[N+](C)(C)C",
        "C(=O)([O-])O",
        "C(F)(Cl)(Br)I",
        "O=P(O)(O)O",
        "CS(=O)(=O)N",
        "C1CCNCC1",
        "c1ccncc1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
        "[O-][N+](=O)c1ccccc1",
    ],
)
def test_rdkit_2d_descriptors_match_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)
    distance_matrix = DistanceMatrix(mol_regular)

    values, feature_names = rdkit_descriptors.calc_2d(mol_regular, distance_matrix)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    expected_values = np.asarray(
        [mordred_values[name] for name in feature_names], dtype=np.float32
    )

    assert feature_names == RDKIT_2D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected_values, rtol=1e-6, atol=1e-6)


def test_calculator_fills_rdkit_2d_descriptor_columns(mordred_2d_calc):
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

    observed = compute(mol, use_3D=False)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    idxs = [FEATURE_NAMES_2D.index(name) for name in RDKIT_2D_FEATURE_NAMES]
    expected_values = np.asarray(
        [mordred_values[name] for name in RDKIT_2D_FEATURE_NAMES], dtype=np.float32
    )

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected_values, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def mordred_all_calc():
    return Calculator(descriptors, ignore_3D=False)


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "c1ccccc1",
        "C(F)(Cl)(Br)I",
        "O=P(O)(O)O",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
    ],
)
def test_rdkit_3d_descriptors_match_mordred(smiles, mordred_all_calc):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    values, feature_names = rdkit_descriptors.calc_3d(mol)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_all_calc.descriptors),
            mordred_all_calc(mol),
            strict=False,
        )
    )
    expected_values = np.asarray(
        [mordred_values[name] for name in feature_names], dtype=np.float32
    )

    assert feature_names == RDKIT_3D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected_values, rtol=1e-6, atol=1e-6)


def test_calculator_fills_rdkit_3d_descriptor_columns(mordred_all_calc):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    observed = compute(mol, use_3D=True)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_all_calc.descriptors),
            mordred_all_calc(mol),
            strict=False,
        )
    )
    idxs = [ALL_FEATURE_NAMES.index(name) for name in RDKIT_3D_FEATURE_NAMES]
    expected_values = np.asarray(
        [mordred_values[name] for name in RDKIT_3D_FEATURE_NAMES], dtype=np.float32
    )

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected_values, rtol=1e-6, atol=1e-6)
