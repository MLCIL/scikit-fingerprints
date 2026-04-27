import importlib

import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["nAromAtom", "nAromBond"]

SMILES = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "C1CCCCC1",
    "C=C",
    "c1ccccc1.CCC",
    "C1CCNCC1",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


def _mordred_expected(mol, mordred_2d_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [mordred_values[name] for name in feature_names], dtype=np.float32
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_aromatic_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    aromatic = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.aromatic"
    )

    values, feature_names = aromatic.calc(cache)
    expected = _mordred_expected(mol, mordred_2d_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_aromatic_columns(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_2d_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)
