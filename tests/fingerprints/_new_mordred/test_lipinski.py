import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import lipinski
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["Lipinski", "GhoseFilter"]

SMILES = [
    "C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCC",
    "CC(=O)[O-].[Na+]",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


def _mordred_values(mol, mordred_2d_calc):
    return dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )


def _expected_lipinski(mol):
    return float(
        rdMolDescriptors.CalcNumHBD(mol) <= 5
        and rdMolDescriptors.CalcNumHBA(mol) <= 10
        and Descriptors.ExactMolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
    )


def _expected_ghose_filter_no_h(mol):
    return float(
        160 <= Descriptors.ExactMolWt(mol) <= 480
        and 20 <= mol.GetNumAtoms() <= 70
        and -0.4 <= Crippen.MolLogP(mol) <= 5.6
        and 40 <= Crippen.MolMR(mol) <= 130
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_lipinski_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = lipinski.calc(cache)
    mordred_values = _mordred_values(mol, mordred_2d_calc)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values[0], mordred_values["Lipinski"], rtol=1e-6, atol=1e-6)
    assert_allclose(values[0], _expected_lipinski(mol), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_ghose_filter_uses_no_h_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = lipinski.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values[1], _expected_ghose_filter_no_h(mol), rtol=1e-6, atol=1e-6)


def test_ghose_filter_differs_from_mordred_default_for_dodecane(mordred_2d_calc):
    mol = Chem.MolFromSmiles("CCCCCCCCCCCC")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, _ = lipinski.calc(cache)
    mordred_values = _mordred_values(mol, mordred_2d_calc)

    assert values[1] == 0.0
    assert mordred_values["GhoseFilter"] == 1


def test_calculator_fills_lipinski_columns():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    assert_equal(observed[idxs].dtype, np.float32)
    assert_allclose(observed[idxs], lipinski.calc(cache)[0], rtol=1e-6, atol=1e-6)
