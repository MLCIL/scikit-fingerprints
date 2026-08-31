import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "smiles, num_atoms, num_h_bond_acceptors, num_h_bond_donors",
    [
        ("CC", 8, 0, 0),
        ("CCO", 9, 1, 1),
        ("c1ccccc1", 12, 0, 0),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", 21, 3, 1),
        ("OC1C(O)C(O)C(O)C(O)C1O", 24, 6, 6),
    ],
)
def test_counts(smiles, num_atoms, num_h_bond_acceptors, num_h_bond_donors):
    mol_regular = preprocess_mol(MolFromSmiles(smiles))

    properties = MolecularProperties(mol_regular)

    assert properties.num_atoms == num_atoms
    assert properties.num_h_bond_acceptors == num_h_bond_acceptors
    assert properties.num_h_bond_donors == num_h_bond_donors


@pytest.mark.parametrize(
    "smiles, log_p, molar_refractivity, exact_mol_wt",
    [
        ("CC", 1.0262, 11.3480, 30.0470),
        ("c1ccccc1", 1.6866, 26.4420, 78.0470),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", 1.3101, 44.7103, 180.0423),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 3.0732, 61.0348, 206.1307),
    ],
)
def test_crippen_and_weight(smiles, log_p, molar_refractivity, exact_mol_wt):
    mol_regular = preprocess_mol(MolFromSmiles(smiles))

    properties = MolecularProperties(mol_regular)

    assert_allclose(properties.log_p, log_p, rtol=1e-4)
    assert_allclose(properties.molar_refractivity, molar_refractivity, rtol=1e-4)
    assert_allclose(properties.exact_mol_wt, exact_mol_wt, rtol=1e-4)
