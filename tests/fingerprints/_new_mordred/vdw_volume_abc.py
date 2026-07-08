import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import vdw_volume_abc
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = ["Vabc"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CC", [43.14843]),
        ("CCO", [51.938656]),
        ("c1ccccc1", [81.166534]),  # 1 aromatic ring
        ("C1CCCCC1", [99.975914]),  # 1 non-aromatic ring
        ("CC(=O)OC1=CC=CC=C1C(=O)O", [162.94247]),
        ("[Na+]", [np.nan]),  # atom without a defined Bondi radius
    ],
)
def test_vdw_volume_abc_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    mol_regular = preprocess_mol(mol)

    values, feature_names = vdw_volume_abc.calc(mol_regular, mol_hydrogens)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
