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

FEATURE_NAMES = [
    "MDEC-11",
    "MDEC-12",
    "MDEC-13",
    "MDEC-14",
    "MDEC-22",
    "MDEC-23",
    "MDEC-24",
    "MDEC-33",
    "MDEC-34",
    "MDEC-44",
    "MDEO-11",
    "MDEO-12",
    "MDEO-22",
    "MDEN-11",
    "MDEN-12",
    "MDEN-13",
    "MDEN-22",
    "MDEN-23",
    "MDEN-33",
]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "c1ccccc1",
    "CCO",
    "CNC(O)O",
    "c1ccncc1",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_molecular_distance_edge_calc():
    return Calculator(descriptors.MolecularDistanceEdge, ignore_3D=True)


def _mordred_expected(mol, mordred_molecular_distance_edge_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_molecular_distance_edge_calc.descriptors),
            mordred_molecular_distance_edge_calc(mol),
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


def test_molecular_distance_edge_feature_names_are_in_mordred_order():
    molecular_distance_edge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_distance_edge"
    )

    assert molecular_distance_edge.FEATURE_NAMES == FEATURE_NAMES
    assert len(molecular_distance_edge.FEATURE_NAMES) == 19


@pytest.mark.parametrize("smiles", SMILES)
def test_molecular_distance_edge_matches_mordred(
    smiles, mordred_molecular_distance_edge_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_distance_edge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_distance_edge"
    )

    values, feature_names = molecular_distance_edge.calc(cache)
    expected = _mordred_expected(
        mol, mordred_molecular_distance_edge_calc, feature_names
    )

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_molecular_distance_edge_columns(
    smiles, mordred_molecular_distance_edge_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(
        mol, mordred_molecular_distance_edge_calc, FEATURE_NAMES
    )

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C", "[Na+].[Cl-]"])
def test_molecular_distance_edge_no_matching_pairs_return_nan_for_all_descriptors(
    smiles,
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_distance_edge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_distance_edge"
    )

    values, _ = molecular_distance_edge.calc(cache)

    assert np.isnan(values).all()


def test_molecular_distance_edge_no_matching_pair_return_nan_for_descriptor():
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_distance_edge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_distance_edge"
    )

    values, feature_names = molecular_distance_edge.calc(cache)

    assert np.isnan(values[feature_names.index("MDEO-11")])
