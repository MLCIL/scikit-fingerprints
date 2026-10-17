import pytest
from mordred._base.context import Context
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

_VARIANTS = [(False, False), (False, True), (True, False), (True, True)]


def _parse_keeping_hydrogens(smiles: str) -> Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


def _kekulized_without_aromatic_flags(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def _hydrogen_first(smiles: str) -> Mol:
    # explicit hydrogens, with one of them moved before every heavy atom
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    hydrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    order = [hydrogen] + [i for i in range(mol.GetNumAtoms()) if i != hydrogen]
    return Chem.RenumberAtoms(mol, order)


def _embedded(smiles: str) -> Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return mol


_INPUTS = {
    "smiles": lambda: Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"),
    "smiles_with_hydrogens_added": lambda: Chem.AddHs(
        Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
    ),
    "kekulized_naphthalene": lambda: _kekulized_without_aromatic_flags(
        "c1ccc2ccccc2c1"
    ),
    "kekulized_imidazole": lambda: _kekulized_without_aromatic_flags("c1cnc[nH]1"),
    "explicit_hydrogens_in_input_order": lambda: _parse_keeping_hydrogens(
        "[H]OC([H])([H])c1ccccc1"
    ),
    "stereo_hydrogen": lambda: _parse_keeping_hydrogens("[H]/N=C/C"),
    "hydrogen_first": lambda: _hydrogen_first("CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    "isotopes": lambda: Chem.MolFromSmiles("[2H]c1ccccc1C"),
    "embedded_conformer": lambda: _embedded("OC(=O)c1ccncc1"),
}


def _signature(mol: Mol) -> tuple[list[tuple], list[tuple]]:
    """Everything the descriptors read from a molecule, in atom and bond order."""
    atoms = [
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetHybridization(),
        )
        for atom in mol.GetAtoms()
    ]
    bonds = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType(),
            bond.GetIsAromatic(),
        )
        for bond in mol.GetBonds()
    ]
    return atoms, bonds


@pytest.mark.parametrize(("explicit_hydrogens", "kekulize"), _VARIANTS)
@pytest.mark.parametrize("input_name", list(_INPUTS))
def test_variants_match_mordred(input_name, explicit_hydrogens, kekulize):
    mol = _INPUTS[input_name]()
    context = Context.from_query(
        mol, False, {explicit_hydrogens}, {kekulize}, -1, config={}
    )
    expected = context._mols[explicit_hydrogens, kekulize]

    preprocessed = preprocess_mol(
        mol, explicit_hydrogens=explicit_hydrogens, kekulize=kekulize
    )

    assert _signature(preprocessed) == _signature(expected)


def test_explicit_hydrogens_do_not_sanitize():
    # RemoveHs would perceive the rings as aromatic again, AddHs keeps them as given
    mol = _kekulized_without_aromatic_flags("c1ccc2ccccc2c1")

    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    mol_regular = preprocess_mol(mol)

    assert not any(atom.GetIsAromatic() for atom in mol_hydrogens.GetAtoms())
    assert all(atom.GetIsAromatic() for atom in mol_regular.GetAtoms())


def test_explicit_hydrogens_keep_atom_order_and_conformers():
    mol = _hydrogen_first("CCO")
    mol.SetIntProp("conf_id", AllChem.EmbedMolecule(mol, randomSeed=0))

    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)

    assert [a.GetAtomicNum() for a in mol_hydrogens.GetAtoms()] == [
        a.GetAtomicNum() for a in mol.GetAtoms()
    ]
    assert mol_hydrogens.GetNumConformers() == 1
    assert mol_hydrogens.GetIntProp("conf_id") == mol.GetIntProp("conf_id")


@pytest.mark.parametrize("explicit_hydrogens", [False, True])
def test_kekulizing_a_variant_keeps_it_and_its_source(explicit_hydrogens):
    variant = preprocess_mol(
        Chem.MolFromSmiles("c1ccccc1O"), explicit_hydrogens=explicit_hydrogens
    )
    num_atoms = variant.GetNumAtoms()

    kekulized = preprocess_mol(variant, kekulize=True, sanitize=False)

    assert kekulized is not variant
    assert kekulized.GetNumAtoms() == num_atoms
    assert any(b.GetBondType() == Chem.BondType.AROMATIC for b in variant.GetBonds())
    assert all(b.GetBondType() != Chem.BondType.AROMATIC for b in kekulized.GetBonds())
