import importlib

import numpy as np
import pytest
from mordred import CPSA, Calculator
from mordred.error import Missing
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

CPSA_FEATURE_NAMES = [
    "PNSA1",
    "PNSA2",
    "PNSA3",
    "PNSA4",
    "PNSA5",
    "PPSA1",
    "PPSA2",
    "PPSA3",
    "PPSA4",
    "PPSA5",
    "DPSA1",
    "DPSA2",
    "DPSA3",
    "DPSA4",
    "DPSA5",
    "FNSA1",
    "FNSA2",
    "FNSA3",
    "FNSA4",
    "FNSA5",
    "FPSA1",
    "FPSA2",
    "FPSA3",
    "FPSA4",
    "FPSA5",
    "WNSA1",
    "WNSA2",
    "WNSA3",
    "WNSA4",
    "WNSA5",
    "WPSA1",
    "WPSA2",
    "WPSA3",
    "WPSA4",
    "WPSA5",
    "RNCG",
    "RPCG",
    "RNCS",
    "RPCS",
    "TASA",
    "TPSA",
    "RASA",
    "RPSA",
]

CPSA_2D_FEATURE_NAMES = ["RNCG", "RPCG"]
CPSA_3D_FEATURE_NAMES = [
    name for name in CPSA_FEATURE_NAMES if name not in CPSA_2D_FEATURE_NAMES
]

SMILES_2D = ["CCO", "C[N+](C)(C)C", "CC(=O)[O-]", "[Na+].[Cl-]", "O"]
SMILES_3D = ["CCO", "c1ccccc1", "C[N+](C)(C)C", "CC(=O)O"]


@pytest.fixture(scope="module")
def mordred_cpsa_calc():
    return Calculator(CPSA, ignore_3D=False)


def _gasteiger_charges(mol):
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return np.asarray(
        [
            atom.GetDoubleProp("_GasteigerCharge")
            + (
                atom.GetDoubleProp("_GasteigerHCharge")
                if atom.HasProp("_GasteigerHCharge")
                else 0.0
            )
            for atom in mol.GetAtoms()
        ],
        dtype=float,
    )


def _relative_charge(charges, positive):
    mask = charges > 0 if positive else charges < 0
    matching_charges = charges[mask]
    if len(matching_charges) == 0:
        return 0.0

    qmax = matching_charges[np.argmax(np.abs(matching_charges))]
    return qmax / np.sum(matching_charges)


def _expected_2d(mol):
    charges = _gasteiger_charges(mol)
    return np.asarray(
        [
            _relative_charge(charges, positive=False),
            _relative_charge(charges, positive=True),
        ],
        dtype=np.float32,
    )


def _mordred_expected(mol, mordred_cpsa_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_cpsa_calc.descriptors),
            mordred_cpsa_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [
            np.nan
            if isinstance(mordred_values[name], Missing)
            else mordred_values[name]
            for name in feature_names
        ],
        dtype=np.float32,
    )


def _embedded_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_cpsa_feature_names_are_in_mordred_order():
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    assert cpsa.FEATURE_NAMES == CPSA_FEATURE_NAMES
    assert cpsa.FEATURE_NAMES_2D == CPSA_2D_FEATURE_NAMES
    assert cpsa.FEATURE_NAMES_3D == CPSA_3D_FEATURE_NAMES
    assert len(cpsa.FEATURE_NAMES) == 43


@pytest.mark.parametrize("smiles", SMILES_2D)
def test_cpsa_2d_matches_local_no_explicit_hydrogen_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    values, feature_names = cpsa.calc_2d(cache)

    assert feature_names == CPSA_2D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, _expected_2d(mol_regular), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", ["[Na+].[Cl-]"])
def test_cpsa_2d_matches_mordred_on_no_h_molecules(smiles, mordred_cpsa_calc):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    values, feature_names = cpsa.calc_2d(cache)
    expected = _mordred_expected(mol_regular, mordred_cpsa_calc, feature_names)

    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES_3D)
def test_cpsa_3d_descriptors_match_mordred(smiles, mordred_cpsa_calc):
    mol = _embedded_mol(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    values, feature_names = cpsa.calc_3d(cache)
    expected = _mordred_expected(mol, mordred_cpsa_calc, feature_names)

    assert feature_names == CPSA_3D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)[O-]"])
def test_calculator_fills_cpsa_2d_columns(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in CPSA_2D_FEATURE_NAMES]

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], _expected_2d(mol_regular), rtol=1e-6, atol=1e-6)


def test_calculator_fills_cpsa_3d_columns(mordred_cpsa_calc):
    mol = _embedded_mol("CCO")

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in CPSA_3D_FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_cpsa_calc, CPSA_3D_FEATURE_NAMES)

    assert_allclose(observed[idxs], expected, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_cpsa_3d_returns_nan_without_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    values, feature_names = cpsa.calc_3d(cache)

    assert feature_names == CPSA_3D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_cpsa_3d_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    cpsa = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.cpsa")

    values, feature_names = cpsa.calc_3d(cache)

    assert feature_names == CPSA_3D_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_fills_cpsa_2d_columns_when_use_3d_is_true():
    mol = _embedded_mol("CCO")
    mol_regular = preprocess_mol(mol)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in CPSA_2D_FEATURE_NAMES]

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], _expected_2d(mol_regular), rtol=1e-6, atol=1e-6)
