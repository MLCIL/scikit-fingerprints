import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import Mol, MolFromSmiles

from skfp.descriptors import constitutional as const


@pytest.fixture
def input_mols() -> dict[str, Mol]:
    result = {}
    for name, smiles in [
        ("oxygen", "O"),
        ("benzene", "C1=CC=CC=C1"),
        ("ethanol", "CCO"),
        ("propane", "CCC"),
        ("butadiene", "C=CC=C"),
        ("hydrogen_cyanide", "C#N"),
        ("cyclohexane", "C1CCCCC1"),
    ]:
        mol = MolFromSmiles(smiles)
        result[name] = mol
    return result


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 13.019),
        ("ethanol", 15.356),
        ("propane", 14.699),
        ("butadiene", 13.523),
        ("hydrogen_cyanide", 13.513),
        ("cyclohexane", 14.027),
        ("oxygen", 18.015),
    ],
)
def test_average_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.average_molecular_weight(mol)
    assert round(result, 3) == round(expected_value, 3)


def test_average_molecular_weight_empty_molecule(input_mols):
    mol = MolFromSmiles("")
    with pytest.raises(
        ValueError,
        match="The molecule must have at least 1 atom.",
    ):
        const.average_molecular_weight(mol)


@pytest.mark.parametrize(
    "mol_name, bond_type, expected_value",
    [
        ("benzene", "SINGLE", 0),
        ("benzene", "DOUBLE", 0),
        ("benzene", "TRIPLE", 0),
        ("benzene", "AROMATIC", 6),
        ("benzene", None, 6),
        ("ethanol", "SINGLE", 2),
        ("ethanol", "DOUBLE", 0),
        ("ethanol", "TRIPLE", 0),
        ("ethanol", None, 2),
        ("butadiene", "SINGLE", 1),
        ("butadiene", "DOUBLE", 2),
        ("butadiene", None, 3),
        ("hydrogen_cyanide", "TRIPLE", 1),
        ("hydrogen_cyanide", None, 1),
        ("cyclohexane", "SINGLE", 6),
        ("cyclohexane", None, 6),
    ],
)
def test_bond_count(mol_name, bond_type, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.bond_count(mol, bond_type)
    assert_equal(result, expected_value)


@pytest.mark.parametrize(
    "mol_name, atom_id, expected_value",
    [
        ("benzene", "C", 6),
        ("ethanol", "O", 1),
        ("propane", "H", 8),
        ("hydrogen_cyanide", "N", 1),
        ("butadiene", 1, 6),
        ("ethanol", 6, 2),
        ("oxygen", 8, 1),
    ],
)
def test_element_atom_count(mol_name, atom_id, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.element_atom_count(mol, atom_id)
    assert_equal(result, expected_value)


@pytest.mark.parametrize(
    "mol_name, atomic_number, expected_value",
    [
        ("benzene", 6, 6),
        ("benzene", 1, 6),
        ("ethanol", 6, 2),
        ("ethanol", 8, 1),
        ("ethanol", 1, 6),
        ("propane", 1, 8),
        ("hydrogen_cyanide", 7, 1),
        ("oxygen", 8, 1),
        ("oxygen", 1, 2),
    ],
)
def test_element_counts(mol_name, atomic_number, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.element_counts(mol)
    assert len(result) == 118
    assert_equal(result[atomic_number - 1], expected_value)


def test_element_counts_matches_element_atom_count(input_mols):
    for mol in input_mols.values():
        counts = const.element_counts(mol)
        assert_equal(counts[5], const.element_atom_count(mol, "C"))
        assert_equal(counts[0], const.element_atom_count(mol, "H"))


def test_element_counts_absent_elements_are_zero(input_mols):
    # benzene contains only carbon and hydrogen
    counts = const.element_counts(input_mols["benzene"])
    assert_equal(counts[7], 0)  # oxygen absent
    assert_equal(counts.sum(), 12)  # 6 C + 6 H


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 6),
        ("ethanol", 3),
        ("propane", 3),
        ("butadiene", 4),
        ("hydrogen_cyanide", 2),
        ("cyclohexane", 6),
        ("oxygen", 1),
    ],
)
def test_heavy_atom_count(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.heavy_atom_count(mol)
    assert_equal(result, expected_value)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 78.114),
        ("ethanol", 46.069),
        ("propane", 44.097),
        ("butadiene", 54.092),
        ("hydrogen_cyanide", 27.026),
        ("cyclohexane", 84.162),
        ("oxygen", 18.015),
    ],
)
def test_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.molecular_weight(mol)
    assert_allclose(result, expected_value, atol=1e-3)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 1),
        ("cyclohexane", 1),
        ("ethanol", 0),
        ("hydrogen_cyanide", 0),
        ("butadiene", 0),
        ("oxygen", 0),
    ],
)
def test_number_of_rings(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_rings(mol)
    assert_equal(result, expected_value)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 0),
        ("ethanol", 0),
        ("propane", 0),
        ("hydrogen_cyanide", 0),
        ("butadiene", 1),
        ("oxygen", 0),
    ],
)
def test_number_of_rotatable_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_rotatable_bonds(mol)
    assert_equal(result, expected_value)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 12),
        ("ethanol", 9),
        ("propane", 11),
        ("butadiene", 10),
        ("hydrogen_cyanide", 3),
        ("cyclohexane", 18),
        ("oxygen", 3),
    ],
)
def test_total_atom_count(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.total_atom_count(mol)
    assert_equal(result, expected_value)
