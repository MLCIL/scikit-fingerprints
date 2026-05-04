import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import acid_base
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["nAcid", "nBase"]

SMILES = [
    "CC(=O)O",
    "CC(=O)[O-]",
    "CS(=O)(=O)O",
    "CN",
    "CNC",
    "CN(C)C",
    "C[NH3+]",
    "c1nn[nH]n1",
    "CCO",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_acid_base_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)

    values, feature_names = acid_base.calc(mol)
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


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_acid_base_columns(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)

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
