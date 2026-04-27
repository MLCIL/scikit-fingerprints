import importlib

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)

GEOMETRICAL_INDEX_FEATURE_NAMES = [
    "GeomDiameter",
    "GeomRadius",
    "GeomShapeIndex",
    "GeomPetitjeanIndex",
]


@pytest.fixture(scope="module")
def mordred_geometrical_index_calc():
    return Calculator(descriptors.GeometricalIndex, ignore_3D=False)


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
            for name in GEOMETRICAL_INDEX_FEATURE_NAMES
        ],
        dtype=np.float32,
    )


def test_geometrical_index_feature_names_are_in_mordred_order():
    geometrical_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.geometrical_index"
    )

    assert geometrical_index.FEATURE_NAMES == GEOMETRICAL_INDEX_FEATURE_NAMES


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_geometrical_index_descriptors_match_mordred(
    smiles, mordred_geometrical_index_calc
):
    mol = _embedded_mol(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    geometrical_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.geometrical_index"
    )

    values, feature_names = geometrical_index.calc(cache)
    expected = _mordred_expected(mol, mordred_geometrical_index_calc)

    assert feature_names == GEOMETRICAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_calculator_fills_geometrical_index_columns(
    smiles, mordred_geometrical_index_calc
):
    mol = _embedded_mol(smiles)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GEOMETRICAL_INDEX_FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_geometrical_index_calc)

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)


def test_geometrical_index_returns_nan_without_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    geometrical_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.geometrical_index"
    )

    values, feature_names = geometrical_index.calc(cache)

    assert feature_names == GEOMETRICAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_geometrical_index_returns_nan_without_conformer():
    mol = Chem.MolFromSmiles("CCO")

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GEOMETRICAL_INDEX_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_geometrical_index_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    geometrical_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.geometrical_index"
    )

    values, feature_names = geometrical_index.calc(cache)

    assert feature_names == GEOMETRICAL_INDEX_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_geometrical_index_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in GEOMETRICAL_INDEX_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_geometrical_index_zero_radius_ratios_are_nan():
    mol = Chem.MolFromSmiles("[He]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    mol.AddConformer(conf)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    geometrical_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.geometrical_index"
    )

    values, feature_names = geometrical_index.calc(cache)

    assert feature_names == GEOMETRICAL_INDEX_FEATURE_NAMES
    assert_allclose(values[:2], [0.0, 0.0], rtol=0, atol=0)
    assert np.isnan(values[2:]).all()


def test_geometrical_index_descriptors_are_3d_only():
    assert all(name not in FEATURE_NAMES_2D for name in GEOMETRICAL_INDEX_FEATURE_NAMES)
