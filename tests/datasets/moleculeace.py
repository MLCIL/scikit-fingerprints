import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.moleculeace import (
    load_chembl204_ki,
    load_chembl214_ki,
    load_chembl218_ec50,
    load_chembl219_ki,
    load_chembl228_ki,
    load_chembl231_ki,
    load_chembl233_ki,
    load_chembl234_ki,
    load_chembl235_ec50,
    load_chembl236_ki,
    load_chembl237_ec50,
    load_chembl237_ki,
    load_chembl238_ki,
    load_chembl239_ec50,
    load_chembl244_ki,
    load_chembl262_ki,
    load_chembl264_ki,
    load_chembl287_ki,
    load_chembl1862_ki,
    load_chembl1871_ki,
    load_chembl2034_ki,
    load_chembl2047_ec50,
    load_chembl2147_ki,
    load_chembl2835_ki,
    load_chembl2971_ki,
    load_chembl3979_ec50,
    load_chembl4005_ki,
    load_chembl4203_ki,
    load_chembl4616_ec50,
    load_chembl4792_ki,
    load_moleculeace_benchmark,
    load_moleculeace_dataset,
    load_moleculeace_splits,
)
from skfp.datasets.moleculeace.benchmark import MOLECULEACE_DATASET_NAMES
from tests.datasets.test_utils import run_basic_dataset_checks


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculeace_benchmark():
    benchmark_full = load_moleculeace_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == MOLECULEACE_DATASET_NAMES

    benchmark_full_tuples = load_moleculeace_benchmark(as_frames=False)
    benchmark_names = [name for name, smiles, y in benchmark_full_tuples]
    assert benchmark_names == MOLECULEACE_DATASET_NAMES


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculeace_benchmark_subset():
    dataset_names = ["chembl4005_ki", "chembl204_ki", "chembl235_ec50"]
    benchmark = load_moleculeace_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark]
    assert benchmark_names == dataset_names


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculeace_benchmark_wrong_subset():
    dataset_names = ["chembl4005_ki", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_moleculeace_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("split_type", ["random", "activity_cliff"])
@pytest.mark.parametrize("dataset_name", MOLECULEACE_DATASET_NAMES)
def test_load_moleculeace_splits(dataset_name, split_type):
    train, test = load_moleculeace_splits(dataset_name, split_type)
    assert isinstance(train, list)
    assert len(train) > 0
    assert all(isinstance(idx, int) for idx in train)

    assert isinstance(test, list)
    assert len(test) > 0
    assert all(isinstance(idx, int) for idx in test)

    assert len(train) > len(test)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("split_type", ["random", "activity_cliff"])
@pytest.mark.parametrize("dataset_name", MOLECULEACE_DATASET_NAMES)
def test_load_moleculeace_splits_as_dict(dataset_name, split_type):
    train, test = load_moleculeace_splits(dataset_name, split_type)
    split_idxs = load_moleculeace_splits(dataset_name, split_type, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["test"] == test


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize(
    "dataset_name, dataset_length",
    [
        ("chembl204_ki", 2754),
        ("chembl214_ki", 3317),
        ("chembl218_ec50", 1031),
        ("chembl219_ki", 1865),
        ("chembl228_ki", 1704),
        ("chembl231_ki", 973),
        ("chembl233_ki", 3142),
        ("chembl234_ki", 3657),
        ("chembl235_ec50", 2349),
        ("chembl236_ki", 2598),
        ("chembl237_ec50", 955),
        ("chembl237_ki", 2603),
        ("chembl238_ki", 1052),
        ("chembl239_ec50", 1721),
        ("chembl244_ki", 3097),
        ("chembl262_ki", 856),
        ("chembl264_ki", 2862),
        ("chembl287_ki", 1328),
        ("chembl1862_ki", 794),
        ("chembl1871_ki", 659),
        ("chembl2034_ki", 750),
        ("chembl2047_ec50", 631),
        ("chembl2147_ki", 1456),
        ("chembl2835_ki", 615),
        ("chembl2971_ki", 976),
        ("chembl3979_ec50", 1125),
        ("chembl4005_ki", 960),
        ("chembl4203_ki", 731),
        ("chembl4616_ec50", 682),
        ("chembl4792_ki", 1471),
    ],
)
def test_load_moleculeace_splits_lengths(dataset_name, dataset_length):
    train, test = load_moleculeace_splits(dataset_name, split_type="random")
    loaded_length = len(train) + len(test)
    assert loaded_length == dataset_length


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("dataset_name", MOLECULEACE_DATASET_NAMES)
def test_load_moleculeace_splits_activity_cliffs(dataset_name):
    random_train, random_test = load_moleculeace_splits(
        dataset_name, split_type="random"
    )
    activity_train, activity_test = load_moleculeace_splits(
        dataset_name, split_type="activity_cliff"
    )

    assert set(random_train) == set(activity_train)
    assert set(random_test) > set(activity_test)


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize("split_type", ["random", "activity_cliff"])
def test_load_moleculeace_splits_nonexistent_dataset(split_type):
    with pytest.raises(InvalidParameterError) as error:
        load_moleculeace_splits("nonexistent", split_type)

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_moleculeace_splits must be a str among"
    )


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
def test_load_moleculeace_splits_nonexistent_splits():
    with pytest.raises(InvalidParameterError) as error:
        load_moleculeace_splits("chembl204_ki", "nonexistent")

    assert str(error.value).startswith(
        "The 'split_type' parameter of load_moleculeace_splits must be a str among"
    )


@pytest.mark.flaky(
    reruns=10,
    reruns_delay=5,
    only_rerun=[LocalEntryNotFoundError, FileNotFoundError],
)
@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("chembl204_ki", load_chembl204_ki, 2754, 1, "regression"),
        ("chembl214_ki", load_chembl214_ki, 3317, 1, "regression"),
        ("chembl218_ec50", load_chembl218_ec50, 1031, 1, "regression"),
        ("chembl219_ki", load_chembl219_ki, 1865, 1, "regression"),
        ("chembl228_ki", load_chembl228_ki, 1704, 1, "regression"),
        ("chembl231_ki", load_chembl231_ki, 973, 1, "regression"),
        ("chembl233_ki", load_chembl233_ki, 3142, 1, "regression"),
        ("chembl234_ki", load_chembl234_ki, 3657, 1, "regression"),
        ("chembl235_ec50", load_chembl235_ec50, 2349, 1, "regression"),
        ("chembl236_ki", load_chembl236_ki, 2598, 1, "regression"),
        ("chembl237_ec50", load_chembl237_ec50, 955, 1, "regression"),
        ("chembl237_ki", load_chembl237_ki, 2603, 1, "regression"),
        ("chembl238_ki", load_chembl238_ki, 1052, 1, "regression"),
        ("chembl239_ec50", load_chembl239_ec50, 1721, 1, "regression"),
        ("chembl244_ki", load_chembl244_ki, 3097, 1, "regression"),
        ("chembl262_ki", load_chembl262_ki, 856, 1, "regression"),
        ("chembl264_ki", load_chembl264_ki, 2862, 1, "regression"),
        ("chembl287_ki", load_chembl287_ki, 1328, 1, "regression"),
        ("chembl1862_ki", load_chembl1862_ki, 794, 1, "regression"),
        ("chembl1871_ki", load_chembl1871_ki, 659, 1, "regression"),
        ("chembl2034_ki", load_chembl2034_ki, 750, 1, "regression"),
        ("chembl2047_ec50", load_chembl2047_ec50, 631, 1, "regression"),
        ("chembl2147_ki", load_chembl2147_ki, 1456, 1, "regression"),
        ("chembl2835_ki", load_chembl2835_ki, 615, 1, "regression"),
        ("chembl2971_ki", load_chembl2971_ki, 976, 1, "regression"),
        ("chembl3979_ec50", load_chembl3979_ec50, 1125, 1, "regression"),
        ("chembl4005_ki", load_chembl4005_ki, 960, 1, "regression"),
        ("chembl4203_ki", load_chembl4203_ki, 731, 1, "regression"),
        ("chembl4616_ec50", load_chembl4616_ec50, 682, 1, "regression"),
        ("chembl4792_ki", load_chembl4792_ki, 1471, 1, "regression"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    df = load_moleculeace_dataset(dataset_name, as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )
