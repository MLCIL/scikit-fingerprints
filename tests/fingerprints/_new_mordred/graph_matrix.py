import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import AddHs, MolFromSmiles

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


def _derived_and_recomputed(mol) -> tuple[DistanceMatrix, DistanceMatrix]:
    mol_hydrogens = AddHs(mol)
    props_hydrogens = AtomicProperties.from_mol(mol_hydrogens)

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
