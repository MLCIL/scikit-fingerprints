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

FEATURE_NAMES = ["fragCpx"]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "c1ccncc1",
    "CC(=O)O",
    "C(F)(Cl)(Br)I",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_fragment_complexity_calc():
    return Calculator(descriptors.FragmentComplexity, ignore_3D=True)


def _as_float(value):
    return np.nan if isinstance(value, Missing) else value


def _mordred_expected(mol, mordred_fragment_complexity_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_fragment_complexity_calc.descriptors),
            mordred_fragment_complexity_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [_as_float(mordred_values[name]) for name in feature_names], dtype=np.float32
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_fragment_complexity_matches_mordred(smiles, mordred_fragment_complexity_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    fragment_complexity = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.fragment_complexity"
    )

    values, feature_names = fragment_complexity.calc(cache)
    expected = _mordred_expected(mol, mordred_fragment_complexity_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_fragment_complexity_column(
    smiles, mordred_fragment_complexity_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_fragment_complexity_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)
