import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from rdkit.Chem import Mol, MolFromFASTA, MolToSmiles

from skfp.preprocessing import MolFromAminoseqTransformer


@pytest.fixture
def sequence_list(fasta_list):
    return [fst.split("\n")[1] for fst in fasta_list]


@pytest.fixture
def peptide_list(fasta_list):
    return [MolFromFASTA(fst) for fst in fasta_list]


@pytest.fixture
def peptides_smiles_list(peptide_list):
    return [MolToSmiles(mol) for mol in peptide_list]


def test_mol_from_fasta(fasta_list):
    mol_from_fasta = MolFromAminoseqTransformer()
    mols_list = mol_from_fasta.transform(fasta_list)

    assert_equal(len(mols_list), len(fasta_list))
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_and_from_fasta(fasta_list, peptides_smiles_list):
    mol_from_fasta = MolFromAminoseqTransformer()

    peptide_list = mol_from_fasta.transform(fasta_list)
    smiles_list = [MolToSmiles(mol) for mol in peptide_list]

    assert_equal(smiles_list, peptides_smiles_list)


def test_mol_from_fasta_and_sequence(fasta_list, sequence_list):
    mol_from_fasta = MolFromAminoseqTransformer()
    mol_from_sequence = MolFromAminoseqTransformer()

    mols_list_fasta = mol_from_fasta.transform(fasta_list)
    mols_list_sequence = mol_from_sequence.transform(sequence_list)

    smiles_list_fasta = [MolToSmiles(mol) for mol in mols_list_fasta]
    smiles_list_sequence = [MolToSmiles(mol) for mol in mols_list_sequence]

    assert_equal(smiles_list_fasta, smiles_list_sequence)


def test_parallel_to_and_from_fasta(peptide_list, fasta_list):
    mol_from_fasta_seq = MolFromAminoseqTransformer()
    mols_list_seq = mol_from_fasta_seq.transform(fasta_list)
    smiles_list_seq = [MolToSmiles(mol) for mol in mols_list_seq]

    mol_from_fasta_parallel = MolFromAminoseqTransformer(n_jobs=-1)
    mols_list_parallel = mol_from_fasta_parallel.transform(fasta_list)
    smiles_list_parallel = [MolToSmiles(mol) for mol in mols_list_parallel]

    assert_equal(smiles_list_seq, smiles_list_parallel)


def test_from_invalid_fasta(fasta_list):
    invalid_fasta_list = ["[H]=[H]", "..."]
    mol_from_aminoseq = MolFromAminoseqTransformer(valid_only=False)
    mols_list = mol_from_aminoseq.transform(fasta_list + invalid_fasta_list)

    mol_from_aminoseq = MolFromAminoseqTransformer(valid_only=True)
    mols_list_2 = mol_from_aminoseq.transform(fasta_list + invalid_fasta_list)

    assert_equal(len(mols_list), len(fasta_list) + len(invalid_fasta_list))
    assert_equal(len(mols_list_2), len(fasta_list))


def test_from_invalid_smiles_with_y(fasta_list):
    invalid_fasta_list = ["[H]=[H]", "..."]
    all_fasta_list = fasta_list + invalid_fasta_list
    labels = np.ones(len(all_fasta_list))
    labels[-len(invalid_fasta_list) :] = 0

    mol_from_aminoseq = MolFromAminoseqTransformer(valid_only=False)
    mols_list, y = mol_from_aminoseq.transform_x_y(all_fasta_list, labels)

    mol_from_aminoseq = MolFromAminoseqTransformer(valid_only=True)
    mols_list_2, y_2 = mol_from_aminoseq.transform_x_y(all_fasta_list, labels)

    assert_equal(len(mols_list), len(all_fasta_list))
    assert_equal(len(mols_list_2), len(fasta_list))

    assert_equal(len(y), len(all_fasta_list))
    assert_equal(len(y_2), len(fasta_list))
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
def test_from_invalid_aminoseq_with_different_label_types(labels):
    sequences = ["AC", "...", "GG"]
    mol_from_aminoseq = MolFromAminoseqTransformer(
        valid_only=True, suppress_warnings=True
    )

    mols, selected_labels = mol_from_aminoseq.transform_x_y(sequences, labels)

    assert_equal(len(mols), 2)
    assert isinstance(selected_labels, type(labels))
    assert_array_equal(np.asarray(selected_labels).ravel(), [10, 30])

    if isinstance(selected_labels, (pd.Series, pd.DataFrame)):
        assert_array_equal(selected_labels.index, [100, 300])
