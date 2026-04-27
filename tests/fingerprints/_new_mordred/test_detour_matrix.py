import importlib

import numpy as np
import pytest
from mordred import Calculator, DetourMatrix
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "DetourIndex",
]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "C1CC2CCC1C2",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_detour_matrix_calc():
    return Calculator(DetourMatrix, ignore_3D=True)


def _mordred_expected(mol, mordred_detour_matrix_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_detour_matrix_calc.descriptors),
            mordred_detour_matrix_calc(mol),
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


def test_detour_matrix_feature_names_are_in_mordred_order():
    detour_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.detour_matrix"
    )

    assert detour_matrix.FEATURE_NAMES == FEATURE_NAMES
    assert len(detour_matrix.FEATURE_NAMES) == 14


@pytest.mark.parametrize("smiles", SMILES)
def test_detour_matrix_matches_mordred(smiles, mordred_detour_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    detour_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.detour_matrix"
    )

    values, feature_names = detour_matrix.calc(cache)
    expected = _mordred_expected(mol, mordred_detour_matrix_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_detour_matrix_columns(smiles, mordred_detour_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_detour_matrix_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C.C", "[Na+].[Cl-]"])
def test_disconnected_molecules_return_nan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    detour_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.detour_matrix"
    )

    values, _ = detour_matrix.calc(cache)

    assert np.isnan(values).all()


def test_one_atom_molecule_has_zero_detour_index(mordred_detour_matrix_calc):
    mol = Chem.MolFromSmiles("C")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    detour_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.detour_matrix"
    )

    values, feature_names = detour_matrix.calc(cache)
    expected = _mordred_expected(mol, mordred_detour_matrix_calc, feature_names)

    assert values[feature_names.index("DetourIndex")] == 0
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_timeout_returns_nan_values():
    mol = Chem.MolFromSmiles("C1CCCCC1")
    cache = MordredMolCache.from_mol(mol, use_3D=False, detour_timeout=-1.0)
    detour_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.detour_matrix"
    )

    values, _ = detour_matrix.calc(cache)

    assert np.isnan(values).all()
