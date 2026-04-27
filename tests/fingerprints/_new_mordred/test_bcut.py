import importlib

import numpy as np
import pytest
from mordred import BCUT, Calculator
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "BCUTc-1h",
    "BCUTc-1l",
    "BCUTdv-1h",
    "BCUTdv-1l",
    "BCUTd-1h",
    "BCUTd-1l",
    "BCUTs-1h",
    "BCUTs-1l",
    "BCUTZ-1h",
    "BCUTZ-1l",
    "BCUTm-1h",
    "BCUTm-1l",
    "BCUTv-1h",
    "BCUTv-1l",
    "BCUTse-1h",
    "BCUTse-1l",
    "BCUTpe-1h",
    "BCUTpe-1l",
    "BCUTare-1h",
    "BCUTare-1l",
    "BCUTp-1h",
    "BCUTp-1l",
    "BCUTi-1h",
    "BCUTi-1l",
]

SMILES = [
    "CC",
    "CCO",
    "c1ccccc1",
    "C(F)(Cl)(Br)I",
    "O=S(=O)(O)O",
    "C.C",
    "[Na+].[Cl-]",
    "O",
    "C[N+](C)(C)C",
]


@pytest.fixture(scope="module")
def mordred_bcut_calc():
    return Calculator(BCUT, ignore_3D=True)


def _mordred_expected(mol, mordred_bcut_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_bcut_calc.descriptors),
            mordred_bcut_calc(mol),
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


def test_bcut_feature_names_are_in_mordred_order():
    bcut = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.bcut")

    assert bcut.FEATURE_NAMES == FEATURE_NAMES
    assert len(bcut.FEATURE_NAMES) == 24


@pytest.mark.parametrize("smiles", SMILES)
def test_bcut_matches_mordred(smiles, mordred_bcut_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    bcut = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.bcut")

    values, feature_names = bcut.calc(cache)
    expected = _mordred_expected(mol, mordred_bcut_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("smiles", ["CCO", "C(F)(Cl)(Br)I"])
def test_calculator_fills_bcut_columns(smiles, mordred_bcut_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_bcut_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-5, atol=1e-5, equal_nan=True)
