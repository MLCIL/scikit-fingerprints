import importlib

import numpy as np
import pytest
from mordred import Calculator, InformationContent
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "IC0",
    "IC1",
    "IC2",
    "IC3",
    "IC4",
    "IC5",
    "TIC0",
    "TIC1",
    "TIC2",
    "TIC3",
    "TIC4",
    "TIC5",
    "SIC0",
    "SIC1",
    "SIC2",
    "SIC3",
    "SIC4",
    "SIC5",
    "BIC0",
    "BIC1",
    "BIC2",
    "BIC3",
    "BIC4",
    "BIC5",
    "CIC0",
    "CIC1",
    "CIC2",
    "CIC3",
    "CIC4",
    "CIC5",
    "MIC0",
    "MIC1",
    "MIC2",
    "MIC3",
    "MIC4",
    "MIC5",
    "ZMIC0",
    "ZMIC1",
    "ZMIC2",
    "ZMIC3",
    "ZMIC4",
    "ZMIC5",
]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "c1ccncc1",
    "CC(=O)O",
    "C.C",
    "[Na+].[Cl-]",
    "[13CH3]CO",
]


@pytest.fixture(scope="module")
def mordred_no_h_information_content_calc():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        InformationContent.InformationContentBase,
        "explicit_hydrogens",
        False,
    )
    try:
        yield Calculator(InformationContent, ignore_3D=True)
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="module")
def mordred_default_information_content_calc():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        InformationContent.InformationContentBase,
        "explicit_hydrogens",
        True,
    )
    try:
        yield Calculator(InformationContent, ignore_3D=True)
    finally:
        monkeypatch.undo()


def _as_float(value):
    return np.nan if isinstance(value, Missing) else value


def _mordred_expected(mol, mordred_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_calc.descriptors),
            mordred_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [_as_float(mordred_values[name]) for name in feature_names],
        dtype=np.float32,
    )


def test_information_content_feature_names_are_in_mordred_order():
    information_content = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.information_content"
    )

    assert information_content.FEATURE_NAMES == FEATURE_NAMES
    assert len(information_content.FEATURE_NAMES) == 42


@pytest.mark.parametrize("smiles", SMILES)
def test_information_content_matches_mordred_no_explicit_hydrogen_policy(
    smiles,
    mordred_no_h_information_content_calc,
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    information_content = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.information_content"
    )

    values, feature_names = information_content.calc(cache)
    expected = _mordred_expected(
        mol,
        mordred_no_h_information_content_calc,
        feature_names,
    )

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C", "[Na+]"])
def test_single_atom_division_cases_return_nan_without_raising(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    information_content = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.information_content"
    )

    values, feature_names = information_content.calc(cache)
    observed = dict(zip(feature_names, values, strict=True))

    assert values.dtype == np.float32
    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isnan([observed[f"SIC{order}"] for order in range(6)]).all()
    assert np.isnan([observed[f"BIC{order}"] for order in range(6)]).all()
    assert_allclose([observed[f"IC{order}"] for order in range(6)], 0, atol=0)
    assert_allclose([observed[f"CIC{order}"] for order in range(6)], 0, atol=0)


@pytest.mark.parametrize("smiles", ["CCO", "c1ccncc1", "[Na+].[Cl-]"])
def test_calculator_fills_information_content_columns(
    smiles,
    mordred_no_h_information_content_calc,
):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(
        mol,
        mordred_no_h_information_content_calc,
        FEATURE_NAMES,
    )

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_default_mordred_differs_from_no_explicit_hydrogen_policy(
    mordred_default_information_content_calc,
):
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    information_content = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.information_content"
    )

    values, feature_names = information_content.calc(cache)
    mordred_values = _mordred_expected(
        mol,
        mordred_default_information_content_calc,
        feature_names,
    )

    matching = np.isclose(values, mordred_values, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert not np.all(matching)
