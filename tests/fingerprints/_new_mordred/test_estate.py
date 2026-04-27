import importlib

import numpy as np
import pytest
from mordred import Calculator, EState
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

ESTATE_ATOM_TYPES = [
    "sLi",
    "ssBe",
    "ssssBe",
    "ssBH",
    "sssB",
    "ssssB",
    "sCH3",
    "dCH2",
    "ssCH2",
    "tCH",
    "dsCH",
    "aaCH",
    "sssCH",
    "ddC",
    "tsC",
    "dssC",
    "aasC",
    "aaaC",
    "ssssC",
    "sNH3",
    "sNH2",
    "ssNH2",
    "dNH",
    "ssNH",
    "aaNH",
    "tN",
    "sssNH",
    "dsN",
    "aaN",
    "sssN",
    "ddsN",
    "aasN",
    "ssssN",
    "sOH",
    "dO",
    "ssO",
    "aaO",
    "sF",
    "sSiH3",
    "ssSiH2",
    "sssSiH",
    "ssssSi",
    "sPH2",
    "ssPH",
    "sssP",
    "dsssP",
    "sssssP",
    "sSH",
    "dS",
    "ssS",
    "aaS",
    "dssS",
    "ddssS",
    "sCl",
    "sGeH3",
    "ssGeH2",
    "sssGeH",
    "ssssGe",
    "sAsH2",
    "ssAsH",
    "sssAs",
    "sssdAs",
    "sssssAs",
    "sSeH",
    "dSe",
    "ssSe",
    "aaSe",
    "dssSe",
    "ddssSe",
    "sBr",
    "sSnH3",
    "ssSnH2",
    "sssSnH",
    "ssssSn",
    "sI",
    "sPbH3",
    "ssPbH2",
    "sssPbH",
    "ssssPb",
]
FEATURE_NAMES = [
    f"{prefix}{atom_type}"
    for prefix in ("N", "S", "MAX", "MIN")
    for atom_type in ESTATE_ATOM_TYPES
]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "c1ccncc1",
    "CS",
    "CP",
    "CCl",
    "CBr",
    "CI",
    "[SiH4]",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_estate_calc():
    return Calculator(EState, ignore_3D=True)


def _mordred_expected(mol, mordred_estate_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_estate_calc.descriptors),
            mordred_estate_calc(mol),
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


def test_estate_feature_names_are_in_mordred_order():
    estate = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.estate"
    )

    assert estate.FEATURE_NAMES == FEATURE_NAMES
    assert len(estate.FEATURE_NAMES) == 316


@pytest.mark.parametrize("smiles", SMILES)
def test_estate_matches_mordred(smiles, mordred_estate_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    estate = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.estate"
    )

    values, feature_names = estate.calc(cache)
    expected = _mordred_expected(mol, mordred_estate_calc, feature_names)

    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_estate_columns(smiles, mordred_estate_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_estate_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_absent_estate_types_have_mordred_missing_policy():
    mol = Chem.MolFromSmiles("C")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    estate = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.estate"
    )

    values, feature_names = estate.calc(cache)
    value_by_name = dict(zip(feature_names, values, strict=True))

    assert value_by_name["NssBe"] == 0
    assert value_by_name["SssBe"] == 0
    assert np.isnan(value_by_name["MAXssBe"])
    assert np.isnan(value_by_name["MINssBe"])
