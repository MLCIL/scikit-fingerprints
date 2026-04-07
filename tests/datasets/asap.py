import pytest
from numpy.testing import assert_equal
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.asap import (
    load_asap_benchmark,
    load_asap_splits,
)
from skfp.datasets.asap.asap import (
    load_hlm,
    load_ksol,
    load_logd,
    load_mdr1_mdckii,
    load_mlm,
    load_pic50_mers_cov,
    load_pic50_sars_cov_2,
)
from skfp.datasets.asap.benchmark import (
    ASAP_DATASET_NAMES,
    _subset_to_dataset_names,
    load_asap_dataset,
)
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_asap_benchmark():
    benchmark_full = load_asap_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, ASAP_DATASET_NAMES)

    benchmark_full_tuples = load_asap_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert_equal(benchmark_names, ASAP_DATASET_NAMES)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_asap_benchmark_subset():
    dataset_names = ["LogD", "KSOL", "MLM"]
    benchmark_full = load_asap_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, dataset_names)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_asap_benchmark_wrong_subset():
    dataset_names = ["LogD", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_asap_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize("dataset_name", ASAP_DATASET_NAMES)
def test_load_asap_splits(dataset_name):
    train, test = load_asap_splits(dataset_name)
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
@pytest.mark.parametrize("dataset_name", ASAP_DATASET_NAMES)
def test_load_asap_splits_as_dict(dataset_name):
    train, test = load_asap_splits(dataset_name)
    split_idxs = load_asap_splits(dataset_name, as_dict=True)
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
        ("HLM", 407),
        ("KSOL", 477),
        ("LogD", 478),
        ("MDR1-MDCKII", 551),
        ("MLM", 425),
        ("pIC50 SARS-CoV-2", 1105),
        ("pIC50 MERS-CoV", 1198),
    ],
)
def test_load_asap_splits_lengths(dataset_name, dataset_length):
    train, test = load_asap_splits(dataset_name)
    loaded_length = len(train) + len(test)
    assert_equal(loaded_length, dataset_length)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_asap_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_asap_splits("nonexistent")

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_asap_splits must be a str among"
    )


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("HLM", load_hlm, 407, 1, "regression"),
        ("KSOL", load_ksol, 477, 1, "regression"),
        ("LogD", load_logd, 478, 1, "regression"),
        ("MDR1-MDCKII", load_mdr1_mdckii, 551, 1, "regression"),
        ("MLM", load_mlm, 425, 1, "regression"),
        (
            "pIC50 SARS-CoV-2",
            load_pic50_sars_cov_2,
            1105,
            1,
            "regression",
        ),
        (
            "pIC50 MERS-CoV",
            load_pic50_mers_cov,
            1198,
            1,
            "regression",
        ),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    # load with load_asap_dataset, to test it simultaneously
    df = load_asap_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
        non_target_columns=["Molecule name"],
    )


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "subset_name, expected_num_datasets",
    [
        (None, 7),
    ],
)
def test_subset_to_dataset_names(subset_name, expected_num_datasets):
    subset_datasets = _subset_to_dataset_names(subset_name)
    assert_equal(len(subset_datasets), expected_num_datasets)


def test_nonexistent_subset_name():
    with pytest.raises(ValueError, match="not recognized"):
        _subset_to_dataset_names(["nonexistent"])
