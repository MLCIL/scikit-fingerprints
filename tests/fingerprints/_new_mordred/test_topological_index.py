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

FEATURE_NAMES = ["Diameter", "Radius", "TopoShapeIndex", "PetitjeanIndex"]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "c1ccccc1",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_topological_index_calc():
    return Calculator(descriptors.TopologicalIndex, ignore_3D=True)


def _mordred_expected(mol, mordred_topological_index_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_topological_index_calc.descriptors),
            mordred_topological_index_calc(mol),
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


def test_topological_index_feature_names_are_in_mordred_order():
    topological_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_index"
    )

    assert topological_index.FEATURE_NAMES == FEATURE_NAMES


@pytest.mark.parametrize("smiles", SMILES)
def test_topological_index_matches_mordred(smiles, mordred_topological_index_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    topological_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_index"
    )

    values, feature_names = topological_index.calc(cache)
    expected = _mordred_expected(mol, mordred_topological_index_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_topological_index_columns(
    smiles, mordred_topological_index_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_topological_index_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C", "[He]", "[Na+]"])
def test_single_atom_ratio_descriptors_are_nan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    topological_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_index"
    )

    values, feature_names = topological_index.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values[:2], [0.0, 0.0], rtol=0, atol=0)
    assert np.isnan(values[2:]).all()


@pytest.mark.parametrize("smiles", ["C.C", "[Na+].[Cl-]"])
def test_disconnected_molecules_keep_rdkit_distance_sentinel(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    topological_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_index"
    )

    values, _ = topological_index.calc(cache)

    assert_allclose(values, [1e8, 1e8, 0.0, 0.0], rtol=0, atol=0)
    assert np.isfinite(values).all()
