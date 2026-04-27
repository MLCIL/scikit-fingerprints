import importlib

import numpy as np
import pytest
from mordred import Autocorrelation, Calculator
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

MAX_DISTANCE = 8
ATS_PROPERTIES = ["dv", "d", "s", "Z", "m", "v", "se", "pe", "are", "p", "i"]
ALL_PROPERTIES = ["c", *ATS_PROPERTIES]

FEATURE_NAMES = [
    *[
        f"ATS{distance}{prop}"
        for prop in ATS_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"AATS{distance}{prop}"
        for prop in ATS_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"ATSC{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"AATSC{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"MATS{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(1, MAX_DISTANCE + 1)
    ],
    *[
        f"GATS{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(1, MAX_DISTANCE + 1)
    ],
]

SMILES = [
    "CC",
    "CCO",
    "c1ccccc1",
    "C(F)(Cl)(Br)I",
    "[Na+].[Cl-]",
    "O",
    "C[N+](C)(C)C",
    "O=S(=O)(O)O",
]


@pytest.fixture
def mordred_autocorrelation_calc(monkeypatch):
    monkeypatch.setattr(
        Autocorrelation.AutocorrelationBase,
        "explicit_hydrogens",
        False,
    )
    return Calculator(Autocorrelation, ignore_3D=True)


def _mordred_expected(mol, mordred_autocorrelation_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_autocorrelation_calc.descriptors),
            mordred_autocorrelation_calc(mol),
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


def test_autocorrelation_feature_names_are_in_mordred_order():
    autocorrelation = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.autocorrelation"
    )

    assert autocorrelation.FEATURE_NAMES == FEATURE_NAMES
    assert len(autocorrelation.FEATURE_NAMES) == 606


@pytest.mark.parametrize("smiles", SMILES)
def test_autocorrelation_matches_no_explicit_hydrogen_mordred(
    smiles, mordred_autocorrelation_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    autocorrelation = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.autocorrelation"
    )

    values, feature_names = autocorrelation.calc(cache)
    expected = _mordred_expected(mol, mordred_autocorrelation_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("smiles", ["CCO", "C(F)(Cl)(Br)I"])
def test_calculator_fills_autocorrelation_columns(smiles, mordred_autocorrelation_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_autocorrelation_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-5, atol=1e-5, equal_nan=True)
