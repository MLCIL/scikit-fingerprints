import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import carbon_types
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = [
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
]

SMILES = [
    "C#CC",
    "C=C",
    "CC=C(C)C",
    "CC(C)(C)C",
    "c1ccccc1",
    "CCO",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_carbon_types_match_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    mol_kekulized = preprocess_mol(mol, explicit_hydrogens=False, kekulize=True)

    values, feature_names = carbon_types.calc(mol_kekulized)
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


def test_calculator_fills_carbon_type_columns(mordred_2d_calc):
    mol = Chem.MolFromSmiles("CC(C)(C)C")

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
