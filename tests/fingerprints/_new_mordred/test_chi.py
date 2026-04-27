import importlib

import numpy as np
import pytest
from mordred import Calculator, Chi
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "Xch-3d",
    "Xch-4d",
    "Xch-5d",
    "Xch-6d",
    "Xch-7d",
    "Xch-3dv",
    "Xch-4dv",
    "Xch-5dv",
    "Xch-6dv",
    "Xch-7dv",
    "Xc-3d",
    "Xc-4d",
    "Xc-5d",
    "Xc-6d",
    "Xc-3dv",
    "Xc-4dv",
    "Xc-5dv",
    "Xc-6dv",
    "Xpc-4d",
    "Xpc-5d",
    "Xpc-6d",
    "Xpc-4dv",
    "Xpc-5dv",
    "Xpc-6dv",
    "Xp-0d",
    "Xp-1d",
    "Xp-2d",
    "Xp-3d",
    "Xp-4d",
    "Xp-5d",
    "Xp-6d",
    "Xp-7d",
    "AXp-0d",
    "AXp-1d",
    "AXp-2d",
    "AXp-3d",
    "AXp-4d",
    "AXp-5d",
    "AXp-6d",
    "AXp-7d",
    "Xp-0dv",
    "Xp-1dv",
    "Xp-2dv",
    "Xp-3dv",
    "Xp-4dv",
    "Xp-5dv",
    "Xp-6dv",
    "Xp-7dv",
    "AXp-0dv",
    "AXp-1dv",
    "AXp-2dv",
    "AXp-3dv",
    "AXp-4dv",
    "AXp-5dv",
    "AXp-6dv",
    "AXp-7dv",
]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "C1CC2CCC1C2",
    "C(F)(Cl)(Br)I",
    "[Na+].[Cl-]",
    "O=S(=O)(O)O",
]


@pytest.fixture(scope="module")
def mordred_chi_calc():
    return Calculator(Chi, ignore_3D=True)


def _mordred_expected(mol, mordred_chi_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_chi_calc.descriptors),
            mordred_chi_calc(mol),
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


def test_chi_feature_names_are_in_mordred_order():
    chi = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.chi")

    assert chi.FEATURE_NAMES == FEATURE_NAMES
    assert len(chi.FEATURE_NAMES) == 56


@pytest.mark.parametrize("smiles", SMILES)
def test_chi_matches_mordred(smiles, mordred_chi_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    chi = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.chi")

    values, feature_names = chi.calc(cache)
    expected = _mordred_expected(mol, mordred_chi_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["CC(C)C", "C1CC2CCC1C2"])
def test_calculator_fills_chi_columns(smiles, mordred_chi_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_chi_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
