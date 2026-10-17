import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles, RenumberAtoms
from rdkit.Chem.AllChem import EmbedMolecule

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol


def _embedded(smiles: str) -> Mol:
    mol = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)
    mol.SetIntProp("conf_id", EmbedMolecule(mol, randomSeed=0))
    return mol


def _hydrogens_moved_to_front(mol: Mol) -> Mol:
    # the same molecule and conformer, with every hydrogen numbered first
    hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    renumbered = RenumberAtoms(mol, hydrogens + heavy)
    renumbered.SetIntProp("conf_id", mol.GetIntProp("conf_id"))
    return renumbered


@pytest.mark.parametrize("use_3D", [False, True])
@pytest.mark.parametrize(
    "smiles", ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)c1ccncc1", "CN1CCC[C@H]1c1cccnc1"]
)
def test_compute_does_not_depend_on_hydrogen_numbering(smiles, use_3D):
    mol = _embedded(smiles)

    values = compute(mol, use_3D)
    values_renumbered = compute(_hydrogens_moved_to_front(mol), use_3D)

    assert_allclose(values_renumbered, values, rtol=1e-4, atol=1e-4, equal_nan=True)
