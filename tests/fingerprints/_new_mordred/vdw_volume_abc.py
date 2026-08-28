import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AddHs

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

    mol_regular = preprocess_mol(mol)
    props_regular = AtomicProperties.from_mol(mol_regular)

    mol_hydrogens = AddHs(mol_regular)
    props_hydrogens = AtomicProperties.with_hydrogens_added(
        mol_hydrogens, props_regular
    )

    rings = RingSets(mol_regular, props_regular)
    values = vdw_volume_abc.calc(rings, props_hydrogens)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
