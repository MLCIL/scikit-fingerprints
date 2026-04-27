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
    "GGI1",
    "GGI2",
    "GGI3",
    "GGI4",
    "GGI5",
    "GGI6",
    "GGI7",
    "GGI8",
    "GGI9",
    "GGI10",
    "JGI1",
    "JGI2",
    "JGI3",
    "JGI4",
    "JGI5",
    "JGI6",
    "JGI7",
    "JGI8",
    "JGI9",
    "JGI10",
    "JGT10",
]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "c1ccccc1",
    "c1ccncc1",
    "CCO",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_topological_charge_calc():
    return Calculator(descriptors.TopologicalCharge, ignore_3D=True)


def _mordred_expected(mol, mordred_topological_charge_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_topological_charge_calc.descriptors),
            mordred_topological_charge_calc(mol),
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


def test_topological_charge_feature_names_are_in_mordred_order():
    topological_charge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_charge"
    )

    assert topological_charge.FEATURE_NAMES == FEATURE_NAMES
    assert len(topological_charge.FEATURE_NAMES) == 21


@pytest.mark.parametrize("smiles", SMILES)
def test_topological_charge_matches_mordred(smiles, mordred_topological_charge_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    topological_charge = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.topological_charge"
    )

    values, feature_names = topological_charge.calc(cache)
    expected = _mordred_expected(mol, mordred_topological_charge_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_topological_charge_columns(
    smiles, mordred_topological_charge_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_topological_charge_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
