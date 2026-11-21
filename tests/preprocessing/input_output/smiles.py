import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSmilesTransformer, MolToSmilesTransformer


def test_mol_from_smiles(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mols_list = mol_from_smiles.transform(smiles_list)

    assert_equal(len(mols_list), len(smiles_list))
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_smiles(mols_list):
    mol_to_smiles = MolToSmilesTransformer()
    smiles_list = mol_to_smiles.transform(mols_list)

    assert_equal(len(smiles_list), len(mols_list))
    assert all(isinstance(x, str) for x in smiles_list)


def test_mol_to_and_from_smiles(mols_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mol_to_smiles = MolToSmilesTransformer()

    smiles_list_1 = mol_to_smiles.transform(mols_list)
    smiles_list_2 = mol_to_smiles.transform(
        mol_from_smiles.transform(mol_to_smiles.transform(mols_list))
    )

    assert smiles_list_1 == smiles_list_2


def test_parallel_to_and_from_smiles(mols_list):
    mol_to_smiles_seq = MolToSmilesTransformer()
    mol_to_smiles_parallel = MolToSmilesTransformer(n_jobs=-1)

    smiles_list_seq = mol_to_smiles_seq.transform(mols_list)
    smiles_list_parallel = mol_to_smiles_parallel.transform(mols_list)

    assert smiles_list_seq == smiles_list_parallel


def test_from_invalid_smiles(smiles_list):
    invalid_smiles_list = ["[H]=[H]", "invalid"]
    mol_from_smiles = MolFromSmilesTransformer(valid_only=False)
    mols_list = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    mol_from_smiles = MolFromSmilesTransformer(valid_only=True)
    mols_list_2 = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    assert_equal(len(mols_list), len(smiles_list) + len(invalid_smiles_list))
    assert_equal(len(mols_list_2), len(smiles_list))


def test_from_invalid_smiles_with_y(smiles_list):
    invalid_smiles_list = ["[H]=[H]", "invalid"]
    all_smiles_list = smiles_list + invalid_smiles_list
    labels = np.ones(len(all_smiles_list))

    labels[-len(invalid_smiles_list) :] = 0

    mol_from_smiles = MolFromSmilesTransformer(valid_only=False)
    mols_list, y = mol_from_smiles.transform_x_y(all_smiles_list, labels)

    mol_from_smiles = MolFromSmilesTransformer(valid_only=True)
    mols_list_2, y_2 = mol_from_smiles.transform_x_y(all_smiles_list, labels)

    assert_equal(len(mols_list), len(all_smiles_list))
    assert_equal(len(mols_list_2), len(smiles_list))

    assert_equal(len(y), len(all_smiles_list))
    assert_equal(len(y_2), len(smiles_list))
    assert_equal(len(mols_list_2), len(y_2))


@pytest.mark.parametrize(
    "labels",
    [
        [10, 20, 30],
        np.array([10, 20, 30]),
        pd.Series([10, 20, 30], index=[100, 200, 300]),
        pd.DataFrame({"label": [10, 20, 30]}, index=[100, 200, 300]),
    ],
)
def test_from_invalid_smiles_with_different_label_types(labels):
    smiles = ["CCO", "invalid", "O"]
    mol_from_smiles = MolFromSmilesTransformer(valid_only=True, suppress_warnings=True)

    mols, selected_labels = mol_from_smiles.transform_x_y(smiles, labels)

    assert_equal(len(mols), 2)
    assert isinstance(selected_labels, type(labels))
    assert_array_equal(np.asarray(selected_labels).ravel(), [10, 30])

    if isinstance(selected_labels, (pd.Series, pd.DataFrame)):
        assert_array_equal(selected_labels.index, [100, 300])
