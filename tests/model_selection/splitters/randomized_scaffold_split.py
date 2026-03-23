import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.randomized_scaffold_split import (
    randomized_scaffold_train_test_split,
    randomized_scaffold_train_valid_test_split,
)
from skfp.model_selection.splitters.scaffold_split import _create_scaffold_sets
from skfp.preprocessing import MolFromSmilesTransformer


@pytest.fixture
def all_molecules() -> list[str]:
    return [
        "CCC",
        "CCCl",
        "CCBr",
        "C1CC1",
        "C1CCC1",
        "C1CCCC1",
        "C1CCCCC1",
        "CCO",
        "CCN",
        "CC.OCC",
        "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]


@pytest.fixture
def smiles_ten_scaffolds() -> list[str]:
    # selected so that they have 10 different scaffolds
    return [
        "C1CCCC(C2CC2)CC1",
        "c1n[nH]cc1C1CCCCCC1",
        "c1n[nH]cc1CC1CCCCCC1",
        "C1CCCC(CC2CCOCC2)CC1",
        "c1ccc2nc(OC3CCC3)ccc2c1",
        "O=C(CCc1cscn1)NC1CCNCC1",
        "c1ccc2nc(OC3CCOC3)ccc2c1",
        "c1ccc2nc(NC3CCOCC3)ccc2c1",
        "c1ccc2nc(N3CCCOCC3)ccc2c1",
        "c1ccc2nc(N3CCn4ccnc4C3)ccc2c1",
    ]


def test_randomized_scaffold_creation_total_count(all_molecules):
    randomized_scaffolds = _create_scaffold_sets(all_molecules)
    assert_equal(len(randomized_scaffolds) <= len(all_molecules), True)


def test_no_ring_molecules():
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
    mols = MolFromSmilesTransformer().transform(smiles)

    randomized_scaffolds = _create_scaffold_sets(mols)
    assert_equal(len(randomized_scaffolds), 1)


def test_randomized_scaffold_count_for_benzodiazepines():
    smiles = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]
    mols = MolFromSmilesTransformer().transform(smiles)

    randomized_scaffolds = _create_scaffold_sets(mols)
    assert_equal(len(randomized_scaffolds), 1)


def test_randomized_scaffold_count_for_xanthines():
    smiles = [
        "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    ]
    mols = MolFromSmilesTransformer().transform(smiles)

    randomized_scaffolds = _create_scaffold_sets(mols)
    assert_equal(len(randomized_scaffolds), 1)


def test_csk_high_degree_atoms():
    smiles = ["O=[U]=O", "F[U](F)(F)(F)(F)F"]
    randomized_scaffold_train_test_split(smiles, random_state=0, use_csk=True)


def test_randomized_scaffold_train_test_split_returns_smiles(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = randomized_scaffold_train_test_split(
        mols, random_state=0, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_randomized_scaffold_train_valid_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, valid_set, test_set = randomized_scaffold_train_valid_test_split(
        mols, random_state=0, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_randomized_scaffold_train_valid_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_idxs, valid_idxs, test_idxs = randomized_scaffold_train_valid_test_split(
        mols, random_state=0, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_seed_consistency_train_test_split(all_molecules):
    train_set_1, test_set_1 = randomized_scaffold_train_test_split(
        all_molecules, random_state=0
    )
    train_set_2, test_set_2 = randomized_scaffold_train_test_split(
        all_molecules, random_state=0
    )

    assert_equal(train_set_1, train_set_2)
    assert_equal(test_set_1, test_set_2)


def test_seed_consistency_train_valid_test_split(all_molecules):
    train_set_1, valid_set_1, test_set_1 = randomized_scaffold_train_valid_test_split(
        all_molecules, random_state=0
    )
    train_set_2, valid_set_2, test_set_2 = randomized_scaffold_train_valid_test_split(
        all_molecules, random_state=0
    )

    assert_equal(train_set_1, train_set_2)
    assert_equal(valid_set_1, valid_set_2)
    assert_equal(test_set_1, test_set_2)


def test_train_test_split_properly_splits_csk_with_ints(smiles_ten_scaffolds):
    train_set, test_set = randomized_scaffold_train_test_split(
        data=smiles_ten_scaffolds, train_size=7, test_size=3, use_csk=True
    )
    assert_equal(len(train_set), 7)
    assert_equal(len(test_set), 3)


def test_train_test_split_properly_splits_csk_with_floats(smiles_ten_scaffolds):
    train_set, test_set = randomized_scaffold_train_test_split(
        data=smiles_ten_scaffolds, train_size=0.7, test_size=0.3, use_csk=True
    )
    assert_equal(len(train_set), 7)
    assert_equal(len(test_set), 3)


def test_train_valid_test_split_properly_splits_csk_with_ints(smiles_ten_scaffolds):
    train_set, valid_set, test_set = randomized_scaffold_train_valid_test_split(
        data=smiles_ten_scaffolds, train_size=7, valid_size=2, test_size=1, use_csk=True
    )
    assert_equal(len(train_set), 7)
    assert_equal(len(valid_set), 2)
    assert_equal(len(test_set), 1)


def test_train_valid_test_split_properly_splits_csk_with_floats(smiles_ten_scaffolds):
    train_set, valid_set, test_set = randomized_scaffold_train_valid_test_split(
        data=smiles_ten_scaffolds,
        train_size=0.7,
        valid_size=0.2,
        test_size=0.1,
        use_csk=True,
    )
    assert_equal(len(train_set), 7)
    assert_equal(len(valid_set), 2)
    assert_equal(len(test_set), 1)


def test_randomized_scaffold_train_test_split_return_indices(all_molecules):
    train_size = 0.6
    test_size = 0.4
    train_idxs, test_idxs = randomized_scaffold_train_test_split(
        all_molecules, train_size=train_size, test_size=test_size, return_indices=True
    )

    assert isinstance(train_idxs, list)
    assert isinstance(test_idxs, list)
    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_train_test_split_with_additional_data(smiles_ten_scaffolds):
    additional_data = list(range(len(smiles_ten_scaffolds)))
    train_set, test_set, train_data, test_data = randomized_scaffold_train_test_split(
        smiles_ten_scaffolds, additional_data
    )
    assert_equal(len(train_set), 8)
    assert_equal(len(test_set), 2)
    assert_equal(len(train_data), 8)
    assert_equal(len(test_data), 2)


def test_train_valid_test_split_with_additional_data(smiles_ten_scaffolds):
    additional_data = list(range(len(smiles_ten_scaffolds)))
    train_set, valid_set, test_set, train_data, valid_data, test_data = (
        randomized_scaffold_train_valid_test_split(
            smiles_ten_scaffolds, additional_data
        )
    )
    assert_equal(len(train_set), 8)
    assert_equal(len(valid_set), 1)
    assert_equal(len(test_set), 1)
    assert_equal(len(train_data), 8)
    assert_equal(len(valid_data), 1)
    assert_equal(len(test_data), 1)


@pytest.mark.parametrize(
    "random_state",
    [0, np.random.RandomState(0), None],
    ids=["int", "RandomState", "None"],
)
def test_randomized_scaffold_split_random_state_types(random_state):
    """randomized_scaffold_train_test_split should accept int, RandomState, or None."""
    smiles = [
        "C1CCCC(C2CC2)CC1",
        "c1n[nH]cc1C1CCCCCC1",
        "c1n[nH]cc1CC1CCCCCC1",
        "C1CCCC(CC2CCOCC2)CC1",
        "c1ccc2nc(OC3CCC3)ccc2c1",
        "O=C(CCc1cscn1)NC1CCNCC1",
        "c1ccc2nc(OC3CCOC3)ccc2c1",
        "c1ccc2nc(NC3CCOCC3)ccc2c1",
        "c1ccc2nc(N3CCCOCC3)ccc2c1",
        "c1ccc2nc(N3CCn4ccnc4C3)ccc2c1",
    ]
    train, test = randomized_scaffold_train_test_split(
        smiles, random_state=random_state
    )
    assert len(train) + len(test) == len(smiles)
