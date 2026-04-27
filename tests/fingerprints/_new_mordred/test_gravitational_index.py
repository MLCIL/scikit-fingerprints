import importlib

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)

GRAVITATIONAL_INDEX_FEATURE_NAMES = ["GRAV", "GRAVH", "GRAVp", "GRAVHp"]


@pytest.fixture(scope="module")
def mordred_gravitational_index_calc():
    return Calculator(descriptors.GravitationalIndex, ignore_3D=False)


def _embedded_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _mordred_expected(mol, mordred_calc):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_calc.descriptors),
            mordred_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [
            np.nan
            if isinstance(mordred_values[name], Missing)
            else mordred_values[name]
            for name in GRAVITATIONAL_INDEX_FEATURE_NAMES
        ],
        dtype=np.float32,
    )


def test_gravitational_index_feature_names_are_in_mordred_order():
    gravitational_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.gravitational_index"
    )

    assert gravitational_index.FEATURE_NAMES == GRAVITATIONAL_INDEX_FEATURE_NAMES


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_gravitational_index_descriptors_match_mordred(
    smiles, mordred_gravitational_index_calc
):
    mol = _embedded_mol(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    gravitational_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.gravitational_index"
    )

    values, feature_names = gravitational_index.calc(cache)
    expected = _mordred_expected(mol, mordred_gravitational_index_calc)

    assert feature_names == GRAVITATIONAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_calculator_fills_gravitational_index_columns(
    smiles, mordred_gravitational_index_calc
):
    mol = _embedded_mol(smiles)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GRAVITATIONAL_INDEX_FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_gravitational_index_calc)

    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)


def test_gravitational_index_returns_nan_without_conformer():
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    gravitational_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.gravitational_index"
    )

    values, feature_names = gravitational_index.calc(cache)

    assert feature_names == GRAVITATIONAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_gravitational_index_returns_nan_without_conformer():
    mol = Chem.MolFromSmiles("CCO")

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GRAVITATIONAL_INDEX_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_gravitational_index_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    gravitational_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.gravitational_index"
    )

    values, feature_names = gravitational_index.calc(cache)

    assert feature_names == GRAVITATIONAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_gravitational_index_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GRAVITATIONAL_INDEX_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_gravitational_index_descriptors_are_3d_only():
    assert all(
        name not in FEATURE_NAMES_2D for name in GRAVITATIONAL_INDEX_FEATURE_NAMES
    )
