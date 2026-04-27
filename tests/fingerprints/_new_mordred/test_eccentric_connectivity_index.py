import importlib

import numpy as np
import pytest
from mordred import Calculator, EccentricConnectivityIndex
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["ECIndex"]

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ccncc1",
    "C.C",
    "CC.O",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_eccentric_connectivity_index_calc():
    return Calculator(EccentricConnectivityIndex, ignore_3D=True)


def _mordred_expected(mol, mordred_eccentric_connectivity_index_calc, feature_names):
    mordred_values = dict(
        zip(
            (
                str(desc)
                for desc in mordred_eccentric_connectivity_index_calc.descriptors
            ),
            mordred_eccentric_connectivity_index_calc(mol),
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


def test_eccentric_connectivity_index_feature_names_are_in_mordred_order():
    eccentric_connectivity_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.eccentric_connectivity_index"
    )

    assert eccentric_connectivity_index.FEATURE_NAMES == FEATURE_NAMES
    assert len(eccentric_connectivity_index.FEATURE_NAMES) == 1


@pytest.mark.parametrize("smiles", SMILES)
def test_eccentric_connectivity_index_matches_mordred(
    smiles, mordred_eccentric_connectivity_index_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eccentric_connectivity_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.eccentric_connectivity_index"
    )

    values, feature_names = eccentric_connectivity_index.calc(cache)
    expected = _mordred_expected(
        mol, mordred_eccentric_connectivity_index_calc, feature_names
    )

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_eccentric_connectivity_index_column(
    smiles, mordred_eccentric_connectivity_index_calc
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(
        mol, mordred_eccentric_connectivity_index_calc, FEATURE_NAMES
    )

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C.C", "CC.O", "[Na+].[Cl-]"])
def test_disconnected_molecules_return_finite_mordred_values(
    smiles, mordred_eccentric_connectivity_index_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eccentric_connectivity_index = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.eccentric_connectivity_index"
    )

    values, feature_names = eccentric_connectivity_index.calc(cache)
    expected = _mordred_expected(
        mol, mordred_eccentric_connectivity_index_calc, feature_names
    )

    assert np.isfinite(expected).all()
    assert np.isfinite(values).all()
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)
