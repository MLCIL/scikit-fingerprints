import pytest
from numpy.testing import assert_equal

from skfp.datasets.lrgb import (
    load_lrgb_mol_benchmark,
    load_lrgb_mol_dataset,
    load_lrgb_mol_splits,
    load_peptides_func,
    load_peptides_struct,
)
from tests.datasets.test_utils import run_basic_dataset_checks


def get_dataset_names() -> list[str]:
    return ["Peptides-func", "Peptides-struct"]


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_lrgb_benchmark():
    dataset_names = get_dataset_names()
    benchmark_full_dfs = load_lrgb_mol_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full_dfs]
    assert_equal(benchmark_names, dataset_names)

    benchmark_full_tuples = load_lrgb_mol_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert_equal(benchmark_names, dataset_names)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_lrgb_splits(dataset_name):
    train, valid, test = load_lrgb_mol_splits(dataset_name)
    assert isinstance(train, list)
    assert len(train) > 0
    assert all(isinstance(idx, int) for idx in train)

    assert isinstance(valid, list)
    assert len(valid) > 0
    assert all(isinstance(idx, int) for idx in valid)

    assert isinstance(test, list)
    assert len(test) > 0
    assert all(isinstance(idx, int) for idx in test)

    assert len(train) > len(valid)
    assert len(train) > len(test)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_lrgb_splits_as_dict(dataset_name):
    train, valid, test = load_lrgb_mol_splits(dataset_name)
    split_idxs = load_lrgb_mol_splits(dataset_name, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "valid", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["valid"] == valid
    assert split_idxs["test"] == test


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, valid_sequences_only, dataset_length",
    [
        ("Peptides-func", False, 15535),
        ("Peptides-struct", False, 15535),
        ("Peptides-func", True, 15424),
        ("Peptides-struct", True, 15424),
    ],
)
def test_load_lrgb_splits_lengths(dataset_name, valid_sequences_only, dataset_length):
    train, valid, test = load_lrgb_mol_splits(dataset_name, valid_sequences_only)
    loaded_length = len(train) + len(valid) + len(test)
    assert_equal(loaded_length, dataset_length)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("Peptides-func", load_peptides_func, 15535, 10, "binary_classification"),
        ("Peptides-struct", load_peptides_struct, 15535, 11, "regression"),
    ],
)
def test_load_dataset_smiles(
    dataset_name, load_func, expected_length, num_tasks, task_type
):
    smiles_list, y = load_func()
    # load with load_lrgb_mol_dataset, to test it simultaneously
    df = load_lrgb_mol_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )
