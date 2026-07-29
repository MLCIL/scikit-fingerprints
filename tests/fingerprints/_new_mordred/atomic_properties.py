import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import AddHs, Atom, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    AtomicProperties,
    get_intrinsic_state,
    get_sigma_electrons,
    get_valence_electrons,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# label -> SMILES; the atom of interest is always index 1
_SMILES = {
    ">C<": "CC(C)(C)C",
    ">CH-": "CC(C)C",
    "-CH2-": "CCC",
    "=C<": "C=C(C)C",
    "-CH3": "CC",
    "=CH-": "CC=C",
    ">N-": "CN(C)C",
    "#C-": "C#CC",
    "-NH-": "CNC",
    "=CH2": "C=C",
    "=N-": "C=NC",
    "-O-": "COC",
    "#CH": "C#C",
    "-NH2": "CN",
    "=NH": "C=N",
    "#N": "C#N",
    "-OH": "CO",
    "=O": "C=O",
    "-F": "CF",
    "-SH": "CS",
    "-S-": "CSC",
    "=S": "C=S",
    "-Cl": "CCl",
    "-Br": "CBr",
    "-I": "CI",
}


def build_atom(label: str, explicit_hs: bool) -> Atom:
    mol = MolFromSmiles(_SMILES[label])
    if explicit_hs:
        mol = AddHs(mol)
    return mol.GetAtomWithIdx(1)


@pytest.mark.parametrize("explicit_hs", [True, False], ids=["explicit_H", "implicit_H"])
@pytest.mark.parametrize(
    ("label", "expected_value"),
    [
        (">C<", 4),
        (">CH-", 3),
        ("-CH2-", 2),
        ("=C<", 3),
        ("-CH3", 1),
        ("=CH-", 2),
        (">N-", 3),
        ("#C-", 2),
        ("-NH-", 2),
        ("=CH2", 1),
        ("=N-", 2),
        ("-O-", 2),
        ("#CH", 1),
        ("-NH2", 1),
        ("=NH", 1),
        ("#N", 1),
        ("-OH", 1),
        ("=O", 1),
        ("-F", 1),
        ("-SH", 1),
        ("-S-", 2),
        ("=S", 1),
        ("-Cl", 1),
        ("-Br", 1),
        ("-I", 1),
    ],
)
def test_sigma_electrons(label, expected_value, explicit_hs):
    atom = build_atom(label, explicit_hs)
    actual_value = get_sigma_electrons(atom)
    assert_allclose(actual_value, expected_value, atol=1e-3)


@pytest.mark.parametrize("explicit_hs", [True, False], ids=["explicit_H", "implicit_H"])
@pytest.mark.parametrize(
    ("label", "expected_value"),
    [
        (">C<", 4),
        (">CH-", 3),
        ("-CH2-", 2),
        ("=C<", 4),
        ("-CH3", 1),
        ("=CH-", 3),
        (">N-", 5),
        ("#C-", 4),
        ("-NH-", 4),
        ("=CH2", 2),
        ("=N-", 5),
        ("-O-", 6),
        ("#CH", 3),
        ("-NH2", 3),
        ("=NH", 4),
        ("#N", 5),
        ("-OH", 5),
        ("=O", 6),
        ("-F", 7),
        ("-S-", 0.67),
        ("-Cl", 0.78),
        ("-Br", 0.26),
        ("-I", 0.16),
    ],
)
def test_valence_electrons(label, expected_value, explicit_hs):
    atom = build_atom(label, explicit_hs)
    actual_value = get_valence_electrons(atom)
    assert_allclose(actual_value, expected_value, atol=1e-2)


@pytest.mark.parametrize("explicit_hs", [True, False], ids=["explicit_H", "implicit_H"])
@pytest.mark.parametrize(
    ("label", "expected_value"),
    [
        (">C<", 1.25),
        (">CH-", 1.3333),
        ("-CH2-", 1.5),
        ("=C<", 1.6666),
        ("-CH3", 2.0),
        ("=CH-", 2.0),
        (">N-", 2.0),
        ("#C-", 2.5),
        ("-NH-", 2.5),
        ("=CH2", 3.0),
        ("=N-", 3.0),
        ("-O-", 3.5),
        ("#CH", 4.0),
        ("-NH2", 4.0),
        ("=NH", 5.0),
        ("#N", 6.0),
        ("-OH", 6.0),
        ("=O", 7.0),
        ("-F", 8.0),
    ],
)
def test_intrinsic_state(label, expected_value, explicit_hs):
    atom = build_atom(label, explicit_hs)
    actual_value = get_intrinsic_state(atom)
    assert_allclose(actual_value, expected_value, atol=1e-3)


# properties that the derived constructors below are expected to fill in
_DERIVED_PROPERTY_NAMES = [
    "atomic_nums",
    "is_hydrogen",
    "is_aromatic",
    "degrees",
    "formal_charges",
    "num_hydrogens",
    "sigma_electrons",
    "valence_electrons",
    "intrinsic_state",
    "bond_begin_idxs",
    "bond_end_idxs",
    "bond_types",
    "bond_orders",
    "bond_is_aromatic",
]

_VARIANT_SMILES = [
    "C",
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "[NH4+]",
    "[2H]OC",
    "F[B-](F)(F)F",
    "CO[13CH3]",
    "c1ccc2[nH]ccc2c1",
    "CC(=O)[O-].[Na+]",
    "[H][H]",
    "O=S(=O)(O)O",
]


@pytest.mark.parametrize("smiles", _VARIANT_SMILES)
@pytest.mark.parametrize("property_name", _DERIVED_PROPERTY_NAMES)
def test_properties_with_hydrogens_added_match_recomputed(smiles, property_name):
    mol = preprocess_mol(MolFromSmiles(smiles))
    mol_hydrogens = AddHs(mol)

    derived = AtomicProperties.with_hydrogens_added(
        mol_hydrogens, AtomicProperties(mol)
    )
    expected = AtomicProperties(mol_hydrogens)

    assert_allclose(
        getattr(derived, property_name), getattr(expected, property_name), rtol=1e-6
    )


@pytest.mark.parametrize("smiles", _VARIANT_SMILES)
@pytest.mark.parametrize("property_name", _DERIVED_PROPERTY_NAMES)
def test_kekulized_properties_match_recomputed(smiles, property_name):
    mol = preprocess_mol(MolFromSmiles(smiles))
    mol_kekulized = preprocess_mol(Mol(mol), kekulize=True, sanitize=False)

    derived = AtomicProperties.kekulized(mol_kekulized, AtomicProperties(mol))
    expected = AtomicProperties(mol_kekulized)

    assert_allclose(
        getattr(derived, property_name), getattr(expected, property_name), rtol=1e-6
    )
