import importlib

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["Kier1", "Kier2", "Kier3"]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ccncc1",
    "C.C",
]


@pytest.fixture(scope="module")
def mordred_kappa_shape_index_calc():
    return Calculator(descriptors.KappaShapeIndex, ignore_3D=True)


def _as_float(value):
    return np.nan if isinstance(value, Missing) else value


def _mordred_expected(mol, mordred_kappa_shape_index_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_kappa_shape_index_calc.descriptors),
            mordred_kappa_shape_index_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [_as_float(mordred_values[name]) for name in feature_names], dtype=np.float32
    )


def test_kappa_shape_index_feature_names_are_in_mordred_order():
    kappa_shape_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.kappa_shape_index"
    )

    assert kappa_shape_index.FEATURE_NAMES == FEATURE_NAMES


@pytest.mark.parametrize("smiles", SMILES)
def test_kappa_shape_index_matches_mordred(smiles, mordred_kappa_shape_index_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    kappa_shape_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.kappa_shape_index"
    )

    values, feature_names = kappa_shape_index.calc(cache)
    expected = _mordred_expected(mol, mordred_kappa_shape_index_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize(
    ("smiles", "nan_idxs"),
    [
        ("C", [0, 1, 2]),
        ("CC", [1, 2]),
        ("CCC", [2]),
    ],
)
def test_kappa_shape_index_returns_nan_when_no_matching_paths(smiles, nan_idxs):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    kappa_shape_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.kappa_shape_index"
    )

    values, _ = kappa_shape_index.calc(cache)

    assert np.isnan(values[nan_idxs]).all()


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_kappa_shape_index_columns(
    smiles, mordred_kappa_shape_index_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_kappa_shape_index_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
