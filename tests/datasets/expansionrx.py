import pytest
from numpy.testing import assert_equal
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.expansionrx import (
    load_expansionrx_benchmark,
    load_expansionrx_splits,
)
from skfp.datasets.expansionrx.benchmark import (
    EXPANSIONRX_DATASET_NAMES,
    _subset_to_dataset_names,
    load_expansionrx_dataset,
)
from skfp.datasets.expansionrx.expansionrx import (
    load_caco2_perm_efflux,
    load_caco2_perm_papp_a_b,
    load_hlm_clint,
    load_ksol,
    load_logd,
    load_mbpb,
    load_mgmb,
    load_mlm_clint,
    load_mppb,
    load_rlm_clint,
)
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_expansionrx_benchmark():
    benchmark_full = load_expansionrx_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, EXPANSIONRX_DATASET_NAMES)

    benchmark_full_tuples = load_expansionrx_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert_equal(benchmark_names, EXPANSIONRX_DATASET_NAMES)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_expansionrx_benchmark_subset():
    dataset_names = ["LogD", "KSOL", "MPPB"]
    benchmark_full = load_expansionrx_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, dataset_names)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_expansionrx_benchmark_wrong_subset():
    dataset_names = ["LogD", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_expansionrx_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize("dataset_name", EXPANSIONRX_DATASET_NAMES)
def test_load_expansionrx_splits(dataset_name):
    train, test = load_expansionrx_splits(dataset_name)
    assert isinstance(train, list)
    assert len(train) > 0
    assert all(isinstance(idx, int) for idx in train)

    assert isinstance(test, list)
    assert len(test) > 0
    assert all(isinstance(idx, int) for idx in test)

    assert len(train) > len(test)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize("dataset_name", EXPANSIONRX_DATASET_NAMES)
def test_load_expansionrx_splits_as_dict(dataset_name):
    train, test = load_expansionrx_splits(dataset_name)
    split_idxs = load_expansionrx_splits(dataset_name, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["test"] == test


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, dataset_length",
    [
        ("LogD", 7331),
        ("KSOL", 7423),
        ("HLM CLint", 4822),
        ("RLM CLint", 633),
        ("MLM CLint", 5805),
        ("Caco-2 Permeability Papp A>B", 3806),
        ("Caco-2 Permeability Efflux", 3803),
        ("MPPB", 1770),
        ("MBPB", 1430),
        ("MGMB", 432),
    ],
)
def test_load_expansionrx_splits_lengths(dataset_name, dataset_length):
    train, test = load_expansionrx_splits(dataset_name)
    loaded_length = len(train) + len(test)
    assert_equal(loaded_length, dataset_length)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_expansionrx_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_expansionrx_splits("nonexistent")

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_expansionrx_splits must be a str among"
    )


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("LogD", load_logd, 7331, 1, "regression"),
        ("KSOL", load_ksol, 7423, 1, "regression"),
        ("HLM CLint", load_hlm_clint, 4822, 1, "regression"),
        ("RLM CLint", load_rlm_clint, 633, 1, "regression"),
        ("MLM CLint", load_mlm_clint, 5805, 1, "regression"),
        (
            "Caco-2 Permeability Papp A>B",
            load_caco2_perm_papp_a_b,
            3806,
            1,
            "regression",
        ),
        (
            "Caco-2 Permeability Efflux",
            load_caco2_perm_efflux,
            3803,
            1,
            "regression",
        ),
        ("MPPB", load_mppb, 1770, 1, "regression"),
        ("MBPB", load_mbpb, 1430, 1, "regression"),
        ("MGMB", load_mgmb, 432, 1, "regression"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    # load with load_expansionrx_dataset, to test it simultaneously
    df = load_expansionrx_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "subset_name, expected_num_datasets",
    [
        (None, 10),
    ],
)
def test_subset_to_dataset_names(subset_name, expected_num_datasets):
    subset_datasets = _subset_to_dataset_names(subset_name)
    assert_equal(len(subset_datasets), expected_num_datasets)


def test_nonexistent_subset_name():
    with pytest.raises(ValueError, match="not recognized"):
        _subset_to_dataset_names(["nonexistent"])
