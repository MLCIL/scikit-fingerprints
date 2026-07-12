import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from numpy.testing import assert_equal
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.moleculenet import (
    load_bace,
    load_bbbp,
    load_clintox,
    load_esol,
    load_freesolv,
    load_hiv,
    load_lipophilicity,
    load_moleculenet_benchmark,
    load_moleculenet_dataset,
    load_muv,
    load_ogb_splits,
    load_pcba,
    load_sider,
    load_tox21,
    load_toxcast,
)
from skfp.datasets.moleculenet.benchmark import (
    MOLECULENET_DATASET_NAMES,
    _subset_to_dataset_names,
)
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculenet_benchmark():
    benchmark_full = load_moleculenet_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, MOLECULENET_DATASET_NAMES)

    benchmark_full_tuples = load_moleculenet_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert_equal(benchmark_names, MOLECULENET_DATASET_NAMES)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculenet_benchmark_subset():
    dataset_names = ["ESOL", "SIDER", "BACE"]
    benchmark_full = load_moleculenet_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, dataset_names)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculenet_benchmark_wrong_subset():
    dataset_names = ["ESOL", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_moleculenet_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("dataset_name", MOLECULENET_DATASET_NAMES)
def test_load_ogb_splits(dataset_name):
    train, valid, test = load_ogb_splits(dataset_name)
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
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("dataset_name", MOLECULENET_DATASET_NAMES)
def test_load_ogb_splits_as_dict(dataset_name):
    train, valid, test = load_ogb_splits(dataset_name)
    split_idxs = load_ogb_splits(dataset_name, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "valid", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["valid"] == valid
    assert split_idxs["test"] == test


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize(
    "dataset_name, dataset_length",
    [
        ("ESOL", 1128),
        ("FreeSolv", 642),
        ("Lipophilicity", 4200),
        ("BACE", 1513),
        ("BBBP", 2039),
        ("HIV", 41127),
        ("ClinTox", 1477),
        ("MUV", 93087),
        ("SIDER", 1427),
        ("Tox21", 7831),
        ("ToxCast", 8576),
        ("PCBA", 437929),
    ],
)
def test_load_ogb_splits_lengths(dataset_name, dataset_length):
    train, valid, test = load_ogb_splits(dataset_name)
    loaded_length = len(train) + len(valid) + len(test)
    assert_equal(loaded_length, dataset_length)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_ogb_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_ogb_splits("nonexistent")

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_ogb_splits must be a str among"
    )


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("ESOL", load_esol, 1128, 1, "regression"),
        ("FreeSolv", load_freesolv, 642, 1, "regression"),
        ("Lipophilicity", load_lipophilicity, 4200, 1, "regression"),
        ("BACE", load_bace, 1513, 1, "binary_classification"),
        ("BBBP", load_bbbp, 2039, 1, "binary_classification"),
        ("HIV", load_hiv, 41127, 1, "binary_classification"),
        ("ClinTox", load_clintox, 1477, 2, "binary_classification"),
        ("MUV", load_muv, 93087, 17, "binary_classification"),
        ("SIDER", load_sider, 1427, 27, "binary_classification"),
        ("Tox21", load_tox21, 7831, 12, "binary_classification"),
        ("ToxCast", load_toxcast, 8576, 617, "binary_classification"),
        ("PCBA", load_pcba, 437929, 128, "binary_classification"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    # load with load_moleculenet_dataset, to test it simultaneously
    df = load_moleculenet_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize(
    "subset_name, expected_num_datasets",
    [
        (None, 12),
        ("classification", 9),
        ("classification_single_task", 3),
        ("classification_multitask", 5),
        ("classification_no_pcba", 8),
        ("regression", 3),
    ],
)
def test_subset_to_dataset_names(subset_name, expected_num_datasets):
    subset_datasets = _subset_to_dataset_names(subset_name)
    assert_equal(len(subset_datasets), expected_num_datasets)


def test_nonexistent_subset_name():
    with pytest.raises(ValueError, match="subset not recognized"):
        _subset_to_dataset_names("nonexistent")
