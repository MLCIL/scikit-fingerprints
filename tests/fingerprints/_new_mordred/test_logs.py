import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import logs
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["FilterItLogS"]

SMILES = [
    "C",
    "CCO",
    "CCN",
    "c1ccncc1",
    "c1ccccc1",
    "c1ccccc1O",
    "CCl",
    "CBr",
    "CI",
    "CC(=O)O",
    "C.C",
]


@pytest.fixture(scope="module")
def mordred_logs_calc():
    return Calculator(descriptors.LogS, ignore_3D=True)


def _as_float(value):
    return np.nan if isinstance(value, Missing) else value


def _mordred_expected(mol, mordred_logs_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_logs_calc.descriptors),
            mordred_logs_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [_as_float(mordred_values[name]) for name in feature_names], dtype=np.float32
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_logs_matches_mordred(smiles, mordred_logs_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = logs.calc(cache)
    expected = _mordred_expected(mol, mordred_logs_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_logs_column(smiles, mordred_logs_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_logs_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)
