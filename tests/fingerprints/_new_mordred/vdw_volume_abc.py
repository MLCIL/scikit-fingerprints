import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import vdw_volume_abc
from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
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

    rings = RingSets(mol_regular, AtomicProperties(mol_regular))
    values = vdw_volume_abc.calc(rings, mol_hydrogens)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
