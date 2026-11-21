import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from rdkit.Chem import Mol, MolFromSmiles, MolToInchi

from skfp.preprocessing import MolFromInchiTransformer, MolToInchiTransformer


@pytest.fixture
def inchi_list(smiles_list):
    return [MolToInchi(MolFromSmiles(smi)) for smi in smiles_list]


def test_mol_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mols_list = mol_from_inchi.transform(inchi_list)

    assert_equal(len(mols_list), len(inchi_list))
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_inchi(mols_list):
    mol_to_inchi = MolToInchiTransformer()
    inchi_list = mol_to_inchi.transform(mols_list)

    assert_equal(len(inchi_list), len(mols_list))
    assert all(isinstance(x, str) for x in inchi_list)


def test_mol_to_and_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mol_to_inchi = MolToInchiTransformer()

    mols_list = mol_from_inchi.transform(inchi_list)
    inchi_list_2 = mol_to_inchi.transform(mols_list)

    assert_equal(inchi_list_2, inchi_list)


def test_parallel_to_and_from_inchi(inchi_list):
    mol_from_inchi_seq = MolFromInchiTransformer()
    mol_from_inchi_parallel = MolFromInchiTransformer(n_jobs=-1)

    mol_to_inchi_seq = MolToInchiTransformer()
    mol_to_inchi_parallel = MolToInchiTransformer(n_jobs=-1)

    mols_list_seq = mol_from_inchi_seq.transform(inchi_list)
    mols_list_parallel = mol_from_inchi_parallel.transform(inchi_list)

    inchi_list_2_seq = mol_to_inchi_seq.transform(mols_list_seq)
    inchi_list_2_parallel = mol_to_inchi_parallel.transform(mols_list_parallel)

    assert_equal(inchi_list_2_seq, inchi_list)
    assert_equal(inchi_list_2_seq, inchi_list_2_parallel)


def test_from_invalid_inchi(inchi_list):
    invalid_inchi_list = ["noninchi", "invalid"]
    mol_from_inchi = MolFromInchiTransformer(valid_only=False)
    mols_list = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols_list_2 = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    assert_equal(len(mols_list), len(inchi_list) + len(invalid_inchi_list))
    assert_equal(len(mols_list_2), len(inchi_list))


def test_inchi_transform_x_y(inchi_list):
    labels = np.ones(len(inchi_list))

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols, labels = mol_from_inchi.transform_x_y(inchi_list, labels)

    assert_equal(len(mols), len(inchi_list))
    assert_equal(len(labels), len(inchi_list))


@pytest.mark.parametrize(
    "labels",
    [
        [10, 20, 30],
        np.array([10, 20, 30]),
        pd.Series([10, 20, 30], index=[100, 200, 300]),
        pd.DataFrame({"label": [10, 20, 30]}, index=[100, 200, 300]),
    ],
)
def test_from_invalid_inchi_with_different_label_types(labels):
    inchi = [
        MolToInchi(MolFromSmiles("CCO")),
        "invalid",
        MolToInchi(MolFromSmiles("O")),
    ]
    mol_from_inchi = MolFromInchiTransformer(valid_only=True, suppress_warnings=True)

    mols, selected_labels = mol_from_inchi.transform_x_y(inchi, labels)

    assert_equal(len(mols), 2)
    assert isinstance(selected_labels, type(labels))
    assert_array_equal(np.asarray(selected_labels).ravel(), [10, 30])

    if isinstance(selected_labels, (pd.Series, pd.DataFrame)):
        assert_array_equal(selected_labels.index, [100, 300])
