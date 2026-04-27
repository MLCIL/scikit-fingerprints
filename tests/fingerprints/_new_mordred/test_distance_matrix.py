import importlib

import numpy as np
import pytest
from mordred import Calculator, DistanceMatrix
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ccncc1",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_distance_matrix_calc():
    return Calculator(DistanceMatrix, ignore_3D=True)


def _mordred_expected(mol, mordred_distance_matrix_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_distance_matrix_calc.descriptors),
            mordred_distance_matrix_calc(mol),
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


def test_distance_matrix_feature_names_are_in_mordred_order():
    distance_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.distance_matrix"
    )

    assert distance_matrix.FEATURE_NAMES == FEATURE_NAMES
    assert len(distance_matrix.FEATURE_NAMES) == 12


@pytest.mark.parametrize("smiles", SMILES)
def test_distance_matrix_matches_mordred(smiles, mordred_distance_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    distance_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.distance_matrix"
    )

    values, feature_names = distance_matrix.calc(cache)
    expected = _mordred_expected(mol, mordred_distance_matrix_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_distance_matrix_columns(smiles, mordred_distance_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_distance_matrix_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C.C", "[Na+].[Cl-]"])
def test_disconnected_molecules_return_nan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    distance_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.distance_matrix"
    )

    values, _ = distance_matrix.calc(cache)

    assert np.isnan(values).all()
