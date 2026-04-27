import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import atom_count
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = [
    "nH",
    "nB",
    "nC",
    "nN",
    "nO",
    "nS",
    "nP",
    "nF",
    "nCl",
    "nBr",
    "nI",
    "nX",
]

SMILES = [
    "CCO",
    "c1ccccc1",
    "C(F)(Cl)(Br)I",
    "O=P(O)(O)O",
    "CS(=O)(=O)N",
    "C[N+](C)(C)C",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_atom_count_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    mol_with_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)

    values, feature_names = atom_count.calc(mol_with_hydrogens)
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


def test_calculator_fills_atom_count_columns(mordred_2d_calc):
    mol = Chem.MolFromSmiles("C(F)(Cl)(Br)I")

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
