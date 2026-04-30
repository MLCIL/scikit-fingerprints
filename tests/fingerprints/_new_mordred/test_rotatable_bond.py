import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import rotatable_bond
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = ["nRot", "RotRatio"]

SMILES = ["CC", "CCCC", "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_rotatable_bond_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    mol_regular = preprocess_mol(mol)

    values, feature_names = rotatable_bond.calc(mol_regular)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    expected = np.asarray(
        [mordred_values[name] for name in feature_names], dtype=np.float32
    )

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_calculator_fills_rotatable_bond_columns(mordred_2d_calc):
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

    observed = compute(mol, use_3D=False)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = np.asarray(
        [mordred_values[name] for name in FEATURE_NAMES], dtype=np.float32
    )

    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)
