import numpy as np
import pytest
from mordred import Calculator, Constitutional
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred import calculator
from skfp.fingerprints._new_mordred.descriptors import constitutional
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol


@pytest.mark.parametrize(
    "smiles",
    ["CCO", "c1ccccc1", "C(F)(Cl)(Br)I", "[Na+]", "[H][H]", ""],
)
def test_constitutional_matches_mordred(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)

    values, feature_names = constitutional.calc(mol_hydrogens)
    mordred_calc = Calculator(Constitutional)
    expected = np.asarray(list(mordred_calc(mol)), dtype=np.float32)

    assert feature_names == [str(desc) for desc in mordred_calc.descriptors]
    assert_allclose(values, expected, rtol=1e-6, equal_nan=True)


def test_constitutional_is_connected_to_calculator():
    mol = Chem.MolFromSmiles("CCO")
    result = calculator.compute(mol, use_3D=False)
    names = calculator.get_feature_names(use_3d=False)
    expected, feature_names = constitutional.calc(
        preprocess_mol(mol, explicit_hydrogens=True)
    )

    indices = [np.flatnonzero(names == name)[0] for name in feature_names]
    assert_allclose(result[indices], expected)
