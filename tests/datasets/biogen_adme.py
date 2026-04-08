import pytest
from numpy.testing import assert_equal

from skfp.datasets.biogen_adme import (
    load_biogen_adme_benchmark,
)
from skfp.datasets.biogen_adme.benchmark import (
    BIOGEN_ADME_DATASET_NAMES,
    _subset_to_dataset_names,
    load_biogen_adme_dataset,
)
from skfp.datasets.biogen_adme.biogen_adme import (
    load_hlm_clint,
    load_hppb,
    load_mdr1_mdck_er,
    load_rlm_clint,
    load_rppb,
    load_solubility,
)
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_biogen_adme_benchmark():
    benchmark_full = load_biogen_adme_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, BIOGEN_ADME_DATASET_NAMES)

    benchmark_full_tuples = load_biogen_adme_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert_equal(benchmark_names, BIOGEN_ADME_DATASET_NAMES)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_biogen_adme_benchmark_subset():
    dataset_names = ["HLM CLint", "Solubility", "RLM CLint"]
    benchmark_full = load_biogen_adme_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert_equal(benchmark_names, dataset_names)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
def test_load_biogen_adme_benchmark_wrong_subset():
    dataset_names = ["HLM CLint", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_biogen_adme_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("HLM CLint", load_hlm_clint, 3087, 1, "regression"),
        ("MDR1-MDCK ER", load_mdr1_mdck_er, 2642, 1, "regression"),
        ("Solubility", load_solubility, 2173, 1, "regression"),
        ("hPPB", load_hppb, 194, 1, "regression"),
        ("rPPB", load_rppb, 168, 1, "regression"),
        ("RLM CLint", load_rlm_clint, 3054, 1, "regression"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    # load with load_biogen_adme_dataset, to test it simultaneously
    df = load_biogen_adme_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
        non_target_columns=["Internal ID"],
    )


@pytest.mark.flaky(
    reruns=100,
    reruns_delay=5,
    only_rerun=["LocalEntryNotFoundError", "FileNotFoundError"],
)
@pytest.mark.parametrize(
    "subset_name, expected_num_datasets",
    [
        (None, 6),
    ],
)
def test_subset_to_dataset_names(subset_name, expected_num_datasets):
    subset_datasets = _subset_to_dataset_names(subset_name)
    assert_equal(len(subset_datasets), expected_num_datasets)


def test_nonexistent_subset_name():
    with pytest.raises(ValueError, match="not recognized"):
        _subset_to_dataset_names(["nonexistent"])
