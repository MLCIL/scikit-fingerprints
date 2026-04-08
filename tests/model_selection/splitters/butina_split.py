import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.butina_split import (
    _create_clusters,
    butina_train_test_split,
    butina_train_valid_test_split,
)
from skfp.preprocessing import MolFromSmilesTransformer


@pytest.fixture
def varied_mols() -> list[str]:
    # diverse set of molecules for clustering
    return [
        "c1ccccc1",  # benzene
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "[Na+].[Cl-]",  # sodium chloride
        "c1ncccc1[C@@H]2CCCN2C",  # nicotine
        "OC2(C(c1ccc(OC)cc1)CN(C)C)CCCCC2",  # venlafaxine
        "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1",  # chlordiazepoxide
        "O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1",  # buspirone
        "CC1OC(C)OC(C)O1",  # paraldehyde
        r"N\1=C(\c3c(Sc2c/1cccc2)cccc3)N4CCN(CCOCCO)CC4",  # quetiapine
    ]


def test_butina_train_test_split_default(varied_mols):
    train, test = butina_train_test_split(varied_mols)
    assert_equal(len(train), 8)
    assert_equal(len(test), 2)


def test_butina_train_test_split_custom_sizes(varied_mols):
    train, test = butina_train_test_split(varied_mols, train_size=0.7, test_size=0.3)
    assert_equal(len(train), 7)
    assert_equal(len(test), 3)


def test_train_test_split_total_molecule_count(varied_mols):
    train_split, test_split = butina_train_test_split(
        varied_mols, train_size=0.8, test_size=0.2
    )
    assert_equal(len(train_split) + len(test_split), len(varied_mols))
    assert_equal(len(train_split), 8)
    assert_equal(len(test_split), 2)


def test_train_valid_test_split_total_molecule_count(varied_mols):
    train_split, valid_split, test_split = butina_train_valid_test_split(
        varied_mols, train_size=0.8, valid_size=0.1, test_size=0.1
    )
    assert_equal(
        len(train_split) + len(valid_split) + len(test_split), len(varied_mols)
    )
    assert_equal(len(train_split), 8)
    assert_equal(len(valid_split), 1)
    assert_equal(len(test_split), 1)


def test_test_split_smaller_than_train_split(varied_mols):
    train_split, test_split = butina_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3
    )
    assert len(train_split) > len(test_split)
    assert_equal(len(train_split), 7)
    assert_equal(len(test_split), 3)


def test_train_split_larger_than_valid_and_test_splits(varied_mols):
    train_split, valid_split, test_split = butina_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.2, test_size=0.1
    )
    assert len(train_split) > len(valid_split) > len(test_split)
    assert_equal(len(train_split), 7)
    assert_equal(len(valid_split), 2)
    assert_equal(len(test_split), 1)


def test_butina_creation_total_count(varied_mols):
    clusters = _create_clusters(varied_mols)
    assert len(clusters) <= len(varied_mols)


def test_butina_count_for_benzodiazepines():
    smiles_list = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]
    mols = MolFromSmilesTransformer().transform(smiles_list)

    clusters = _create_clusters(mols)
    assert_equal(len(clusters), 1)


def test_butina_train_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train_set, test_set = butina_train_test_split(mols, return_indices=False)

    assert all(isinstance(m, Mol) for m in train_set)
    assert all(isinstance(m, Mol) for m in test_set)


def test_butina_train_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train_idxs, test_idxs = butina_train_test_split(mols, return_indices=True)

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_butina_train_valid_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train_set, valid_set, test_set = butina_train_valid_test_split(
        mols, return_indices=False
    )

    assert all(isinstance(m, Mol) for m in train_set)
    assert all(isinstance(m, Mol) for m in valid_set)
    assert all(isinstance(m, Mol) for m in test_set)


def test_butina_train_valid_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train_idxs, valid_idxs, test_idxs = butina_train_valid_test_split(
        mols, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_butina_train_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    labels = np.ones(len(varied_mols))
    train_mols, test_mols, train_labels, test_labels = butina_train_test_split(
        mols, labels
    )

    assert_equal(len(train_mols), len(train_labels))
    assert_equal(len(test_mols), len(test_labels))


def test_butina_train_valid_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    labels = np.ones(len(varied_mols))
    (
        train_mols,
        valid_mols,
        test_mols,
        train_labels,
        valid_labels,
        test_labels,
    ) = butina_train_valid_test_split(mols, labels)

    assert_equal(len(train_mols), len(train_labels))
    assert_equal(len(valid_mols), len(valid_labels))
    assert_equal(len(test_mols), len(test_labels))


def test_empty_train_subset_raises_an_error_train_test():
    smiles_list = ["C1CCCC(C2CC2)CC1"]
    with pytest.raises(ValueError, match="the resulting train set will be empty"):
        butina_train_test_split(data=smiles_list)


def test_empty_train_subset_raises_an_error_train_valid_test():
    smiles_list = ["C1CCCC(C2CC2)CC1", "c1n[nH]cc1C1CCCCCC1"]
    with pytest.raises(ValueError, match="one of the sets will be empty"):
        butina_train_valid_test_split(data=smiles_list)
