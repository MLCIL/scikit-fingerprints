import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles, RenumberAtoms

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol


@pytest.mark.parametrize(
    "smiles",
    [
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
    ],
)
def test_distances_with_hydrogens_added_match_recomputed(smiles):
    mol = preprocess_mol(MolFromSmiles(smiles))
    derived, expected = _derived_and_recomputed(mol)
    _assert_matrices_equal(derived, expected)


def test_distances_with_hydrogens_added_match_recomputed_reference_mols(
    mordred_test_mols,
):
    for name, mol in mordred_test_mols.items():
        derived, expected = _derived_and_recomputed(preprocess_mol(mol))
        try:
            _assert_matrices_equal(derived, expected)
        except AssertionError as err:
            raise AssertionError(f"distances differ for molecule {name}") from err


def _hydrogens_reordered(smiles: str, hydrogen_position: int) -> Mol:
    # explicit hydrogens, with the last one moved to the given position
    mol = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)
    order = list(range(mol.GetNumAtoms() - 1))
    order.insert(hydrogen_position, mol.GetNumAtoms() - 1)
    return RenumberAtoms(mol, order)


@pytest.mark.parametrize(
    ("smiles", "hydrogen_position"),
    [
        ("CC(=O)Oc1ccccc1C(=O)O", 0),
        ("c1ccc2[nH]ccc2c1", 4),
        # after every heavy atom, but not in the order AddHs appends them
        ("OCC(O)CO", 6),
        ("CC(=O)[O-].[Na+]", 1),
    ],
)
def test_distances_with_hydrogens_added_to_the_input_match_recomputed(
    smiles, hydrogen_position
):
    # the calculator adds hydrogens to the input, not to the RemoveHs molecule
    mol = _hydrogens_reordered(smiles, hydrogen_position)
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)

    derived = DistanceMatrix.with_hydrogens_added(
        DistanceMatrix.from_mol(preprocess_mol(mol)),
        AtomicProperties(mol_hydrogens),
    )
    expected = DistanceMatrix.from_mol(mol_hydrogens)
    _assert_matrices_equal(derived, expected)


def _derived_and_recomputed(mol) -> tuple[DistanceMatrix, DistanceMatrix]:
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    props_hydrogens = AtomicProperties(mol_hydrogens)

    derived = DistanceMatrix.with_hydrogens_added(
        DistanceMatrix.from_mol(mol), props_hydrogens
    )
    expected = DistanceMatrix.from_mol(mol_hydrogens)
    return derived, expected


def _assert_matrices_equal(derived: DistanceMatrix, expected: DistanceMatrix) -> None:
    assert_allclose(derived.matrix, expected.matrix)
    assert_allclose(derived.eccentricities, expected.eccentricities)
    assert_allclose(derived.radius, expected.radius)
    assert_allclose(derived.diameter, expected.diameter)
