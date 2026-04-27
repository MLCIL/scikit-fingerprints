import importlib

import numpy as np
import pytest
from mordred import BaryszMatrix, Calculator
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

PROPERTIES = ["Z", "m", "v", "se", "pe", "are", "p", "i"]
ATTRIBUTES = [
    "SpAbs",
    "SpMax",
    "SpDiam",
    "SpAD",
    "SpMAD",
    "LogEE",
    "SM1",
    "VE1",
    "VE2",
    "VE3",
    "VR1",
    "VR2",
    "VR3",
]

FEATURE_NAMES = [f"{attr}_Dz{prop}" for prop in PROPERTIES for attr in ATTRIBUTES]

SMILES = [
    "CC",
    "CCO",
    "c1ccccc1",
    "C(F)(Cl)(Br)I",
    "O=S(=O)(O)O",
    "C.C",
    "[Na+].[Cl-]",
    "O",
]


@pytest.fixture(scope="module")
def mordred_barysz_matrix_calc():
    return Calculator(BaryszMatrix, ignore_3D=True)


def _mordred_expected(mol, mordred_barysz_matrix_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_barysz_matrix_calc.descriptors),
            mordred_barysz_matrix_calc(mol),
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


def test_barysz_matrix_feature_names_are_in_mordred_order():
    barysz_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.barysz_matrix"
    )

    assert barysz_matrix.FEATURE_NAMES == FEATURE_NAMES
    assert len(barysz_matrix.FEATURE_NAMES) == 104


@pytest.mark.parametrize("smiles", SMILES)
def test_barysz_matrix_matches_mordred(smiles, mordred_barysz_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    barysz_matrix = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.barysz_matrix"
    )

    values, feature_names = barysz_matrix.calc(cache)
    expected = _mordred_expected(mol, mordred_barysz_matrix_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("smiles", ["CCO", "C(F)(Cl)(Br)I"])
def test_calculator_fills_barysz_matrix_columns(smiles, mordred_barysz_matrix_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_barysz_matrix_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-5, atol=1e-5, equal_nan=True)
