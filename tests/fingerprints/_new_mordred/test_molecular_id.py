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
    "MID",
    "AMID",
    "MID_h",
    "AMID_h",
    "MID_C",
    "AMID_C",
    "MID_N",
    "AMID_N",
    "MID_O",
    "AMID_O",
    "MID_X",
    "AMID_X",
]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "c1ccncc1",
    "CCl",
    "CBr",
    "C(F)(Cl)(Br)I",
    "CCN",
    "C.C",
    "[Na+].[Cl-]",
]


@pytest.fixture(scope="module")
def mordred_molecular_id_calc():
    return Calculator(descriptors.MolecularId, ignore_3D=True)


def _mordred_expected(mol, mordred_molecular_id_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_molecular_id_calc.descriptors),
            mordred_molecular_id_calc(mol),
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


def test_molecular_id_feature_names_are_in_mordred_order():
    molecular_id = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_id"
    )

    assert molecular_id.FEATURE_NAMES == FEATURE_NAMES
    assert len(molecular_id.FEATURE_NAMES) == 12


@pytest.mark.parametrize("smiles", SMILES)
def test_molecular_id_matches_mordred(smiles, mordred_molecular_id_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_id = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_id"
    )

    values, feature_names = molecular_id.calc(cache)
    expected = _mordred_expected(mol, mordred_molecular_id_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_molecular_id_columns(smiles, mordred_molecular_id_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_molecular_id_calc, FEATURE_NAMES)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", ["C.C", "[Na+].[Cl-]"])
def test_molecular_id_disconnected_molecules_return_nan_for_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_id = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_id"
    )

    values, _ = molecular_id.calc(cache)

    assert np.isnan(values).all()


def test_molecular_id_empty_molecule_returns_nan_for_all_descriptors():
    cache = MordredMolCache.from_mol(Chem.Mol(), use_3D=False)
    molecular_id = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_id"
    )

    values, _ = molecular_id.calc(cache)

    assert np.isnan(values).all()


def test_molecular_id_averaged_descriptors_divide_by_total_atom_count():
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    molecular_id = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.molecular_id"
    )

    values, feature_names = molecular_id.calc(cache)

    assert values[feature_names.index("AMID_O")] == pytest.approx(
        values[feature_names.index("MID_O")] / mol.GetNumAtoms()
    )
