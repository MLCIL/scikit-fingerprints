import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import mcgowan_volume
from skfp.fingerprints._new_mordred.utils.atomic_properties import get_mc_gowan_volume
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["VMcGowan"]

SMILES = [
    "C",
    "CC",
    "CCO",
    "c1ccccc1",
    "[NH4+]",
    "[Na+].[Cl-]",
    "[Xe]",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


def _expected_mcgowan_volume_no_h(mol):
    return (
        sum(get_mc_gowan_volume(atom) for atom in mol.GetAtoms())
        - 6.56 * mol.GetNumBonds()
    )


def _mordred_values(mol, mordred_2d_calc):
    return dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_mcgowan_volume_uses_no_h_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = mcgowan_volume.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, [_expected_mcgowan_volume_no_h(mol)], rtol=1e-6, atol=1e-6)


def test_mcgowan_volume_returns_nan_for_unsupported_element():
    mol = Chem.MolFromSmiles("[Og]")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = mcgowan_volume.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values[0])


@pytest.mark.parametrize("smiles", ["C", "CCO"])
def test_mcgowan_volume_differs_from_mordred_default(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, _ = mcgowan_volume.calc(cache)
    mordred_values = _mordred_values(mol, mordred_2d_calc)

    assert_allclose(values, [_expected_mcgowan_volume_no_h(mol)], rtol=1e-6, atol=1e-6)
    assert values[0] != mordred_values["VMcGowan"]


def test_calculator_fills_mcgowan_volume_column():
    mol = Chem.MolFromSmiles("CCO")

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    assert_equal(observed[idxs].dtype, np.float32)
    assert_allclose(
        observed[idxs],
        mcgowan_volume.calc(cache)[0],
        rtol=1e-6,
        atol=1e-6,
    )
