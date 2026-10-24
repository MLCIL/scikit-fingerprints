import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
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
    "mol_name, expected_counts",
    [
        ("oxygen", {1: 2, 8: 1}),
        ("benzene", {1: 6, 6: 6}),
        ("ethanol", {1: 6, 6: 2, 8: 1}),
        ("propane", {1: 8, 6: 3}),
        ("butadiene", {1: 6, 6: 4}),
        ("hydrogen_cyanide", {1: 1, 6: 1, 7: 1}),
        ("cyclohexane", {1: 12, 6: 6}),
    ],
)
def test_element_counts(mol_name, expected_counts, input_mols):
    mol = input_mols[mol_name]
    result = const.element_counts(mol, include_hydrogens=True)

    expected = np.zeros(118, dtype=np.uint32)

    for atomic_num, count in expected_counts.items():
        expected[atomic_num - 1] = count

    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "smiles, expected_counts",
    [
        ("[H][H]", {1: 2}),
        ("[2H]O[2H]", {1: 2, 8: 1}),
        ("[NH4+]", {1: 4, 7: 1}),
        ("[Na+].[Cl-]", {11: 1, 17: 1}),
        ("[Og]", {118: 1}),
        ("*", {}),
        ("", {}),
    ],
)
def test_element_counts_special_cases(smiles, expected_counts):
    mol = MolFromSmiles(smiles)
    result = const.element_counts(mol, include_hydrogens=True)

    expected = np.zeros(118, dtype=np.uint32)

    for atomic_num, count in expected_counts.items():
        expected[atomic_num - 1] = count

    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "mol_name, expected_counts",
    [
        ("oxygen", {8: 1}),
        ("benzene", {6: 6}),
        ("ethanol", {6: 2, 8: 1}),
        ("propane", {6: 3}),
        ("butadiene", {6: 4}),
        ("hydrogen_cyanide", {6: 1, 7: 1}),
        ("cyclohexane", {6: 6}),
    ],
)
def test_element_counts_without_hydrogens(mol_name, expected_counts, input_mols):
    mol = input_mols[mol_name]
    result = const.element_counts(mol)

    expected = np.zeros(118, dtype=np.uint32)
    for atomic_num, count in expected_counts.items():
        expected[atomic_num - 1] = count

    assert_array_equal(result, expected)


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
