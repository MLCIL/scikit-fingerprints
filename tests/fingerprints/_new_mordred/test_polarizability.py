import importlib

import numpy as np
import pytest
from mordred import Calculator, Polarizability
from mordred.error import Missing
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.atomic_properties import get_polarizability
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["apol", "bpol"]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "CCl",
    "[Na+].[Cl-]",
    "[Xe]",
]


@pytest.fixture(scope="module")
def mordred_polarizability_calc():
    return Calculator(Polarizability, ignore_3D=True)


def _expected_no_explicit_h(mol):
    try:
        apol = sum(get_polarizability(atom) for atom in mol.GetAtoms())
        bpol = sum(
            abs(
                get_polarizability(bond.GetBeginAtom())
                - get_polarizability(bond.GetEndAtom())
            )
            for bond in mol.GetBonds()
        )
    except (IndexError, KeyError):
        apol = np.nan
        bpol = np.nan

    return np.asarray([apol, bpol], dtype=np.float32)


def _mordred_expected(mol, mordred_polarizability_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_polarizability_calc.descriptors),
            mordred_polarizability_calc(mol),
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


def test_polarizability_feature_names_are_in_mordred_order():
    polarizability = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.polarizability"
    )

    assert polarizability.FEATURE_NAMES == FEATURE_NAMES


@pytest.mark.parametrize("smiles", SMILES)
def test_polarizability_matches_local_no_explicit_hydrogen_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    polarizability = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.polarizability"
    )

    values, feature_names = polarizability.calc(cache)
    expected = _expected_no_explicit_h(mol)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_polarizability_returns_nan_for_unsupported_element():
    mol = Chem.MolFromSmiles("[Og]C")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    polarizability = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.polarizability"
    )

    values, feature_names = polarizability.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


@pytest.mark.parametrize("smiles", ["C", "CCO"])
def test_polarizability_default_mordred_uses_explicit_hydrogens(
    smiles,
    mordred_polarizability_calc,
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    polarizability = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.polarizability"
    )

    values, feature_names = polarizability.calc(cache)
    mordred_values = _mordred_expected(mol, mordred_polarizability_calc, feature_names)

    assert_allclose(values, _expected_no_explicit_h(mol), rtol=1e-6, atol=1e-6)
    assert not np.allclose(values, mordred_values, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", ["CCO", "CCl", "[Na+].[Cl-]"])
def test_calculator_fills_polarizability_columns(smiles):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _expected_no_explicit_h(mol)

    assert_equal(observed[idxs].dtype, np.float32)
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
