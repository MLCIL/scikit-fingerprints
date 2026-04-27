import importlib

import numpy as np
import pytest
from mordred import Calculator, Constitutional
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_allred_rocow_en,
    get_ionization_potential,
    get_mass,
    get_pauling_en,
    get_polarizability,
    get_sanderson_en,
    get_vdw_volume,
)
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "SZ",
    "Sm",
    "Sv",
    "Sse",
    "Spe",
    "Sare",
    "Sp",
    "Si",
    "MZ",
    "Mm",
    "Mv",
    "Mse",
    "Mpe",
    "Mare",
    "Mp",
    "Mi",
]

SMILES = [
    "C",
    "CCO",
    "c1ccccc1",
    "C(F)(Cl)(Br)I",
    "[Na+].[Cl-]",
    "O=S(=O)(O)O",
    "C[N+](C)(C)C",
]

PROPERTIES = [
    ("Z", lambda atom: atom.GetAtomicNum()),
    ("m", get_mass),
    ("v", get_vdw_volume),
    ("se", get_sanderson_en),
    ("pe", get_pauling_en),
    ("are", get_allred_rocow_en),
    ("p", get_polarizability),
    ("i", get_ionization_potential),
]


@pytest.fixture(scope="module")
def mordred_constitutional_calc():
    return Calculator(Constitutional, ignore_3D=True)


def _expected_no_explicit_h(mol):
    atoms = list(mol.GetAtoms())
    carbon = Chem.MolFromSmiles("C").GetAtomWithIdx(0)

    sums = []
    for _, prop_func in PROPERTIES:
        carbon_value = prop_func(carbon)
        values = np.asarray([prop_func(atom) for atom in atoms], dtype=float)
        sums.append(np.sum(values / carbon_value))

    means = [value / mol.GetNumAtoms() for value in sums]
    return np.asarray([*sums, *means], dtype=np.float32)


def _mordred_expected(mol, mordred_constitutional_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_constitutional_calc.descriptors),
            mordred_constitutional_calc(mol),
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


def test_constitutional_feature_names_are_in_mordred_order():
    constitutional = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.constitutional"
    )

    assert constitutional.FEATURE_NAMES == FEATURE_NAMES
    assert len(constitutional.FEATURE_NAMES) == 16


@pytest.mark.parametrize("smiles", SMILES)
def test_constitutional_matches_local_no_explicit_hydrogen_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    constitutional = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.constitutional"
    )

    values, feature_names = constitutional.calc(cache)
    expected = _expected_no_explicit_h(mol)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_constitutional_default_mordred_uses_explicit_hydrogens(
    mordred_constitutional_calc,
):
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    constitutional = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.constitutional"
    )

    values, feature_names = constitutional.calc(cache)
    mordred_values = _mordred_expected(mol, mordred_constitutional_calc, feature_names)
    matching = np.isclose(values, mordred_values, rtol=1e-6, atol=1e-6, equal_nan=True)

    assert not np.all(matching)
    assert_allclose(
        values[matching],
        mordred_values[matching],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_constitutional_empty_molecule_returns_nan_means():
    mol = Chem.MolFromSmiles("")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    constitutional = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.constitutional"
    )

    values, feature_names = constitutional.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values[:8], np.zeros(8, dtype=np.float32), rtol=0, atol=0)
    assert np.isnan(values[8:]).all()


@pytest.mark.parametrize("smiles", ["CCO", "C(F)(Cl)(Br)I"])
def test_calculator_fills_constitutional_columns(smiles):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _expected_no_explicit_h(mol)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
