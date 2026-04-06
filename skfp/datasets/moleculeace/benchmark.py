import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .moleculeace import (
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
)

MOLECULEACE_DATASET_NAMES = [
    "chembl204_ki",
    "chembl214_ki",
    "chembl218_ec50",
    "chembl219_ki",
    "chembl228_ki",
    "chembl231_ki",
    "chembl233_ki",
    "chembl234_ki",
    "chembl235_ec50",
    "chembl236_ki",
    "chembl237_ec50",
    "chembl237_ki",
    "chembl238_ki",
    "chembl239_ec50",
    "chembl244_ki",
    "chembl262_ki",
    "chembl264_ki",
    "chembl287_ki",
    "chembl1862_ki",
    "chembl1871_ki",
    "chembl2034_ki",
    "chembl2047_ec50",
    "chembl2147_ki",
    "chembl2835_ki",
    "chembl2971_ki",
    "chembl3979_ec50",
    "chembl4005_ki",
    "chembl4203_ki",
    "chembl4616_ec50",
    "chembl4792_ki",
]

MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC = {
    "chembl204_ki": load_chembl204_ki,
    "chembl214_ki": load_chembl214_ki,
    "chembl218_ec50": load_chembl218_ec50,
    "chembl219_ki": load_chembl219_ki,
    "chembl228_ki": load_chembl228_ki,
    "chembl231_ki": load_chembl231_ki,
    "chembl233_ki": load_chembl233_ki,
    "chembl234_ki": load_chembl234_ki,
    "chembl235_ec50": load_chembl235_ec50,
    "chembl236_ki": load_chembl236_ki,
    "chembl237_ec50": load_chembl237_ec50,
    "chembl237_ki": load_chembl237_ki,
    "chembl238_ki": load_chembl238_ki,
    "chembl239_ec50": load_chembl239_ec50,
    "chembl244_ki": load_chembl244_ki,
    "chembl262_ki": load_chembl262_ki,
    "chembl264_ki": load_chembl264_ki,
    "chembl287_ki": load_chembl287_ki,
    "chembl1862_ki": load_chembl1862_ki,
    "chembl1871_ki": load_chembl1871_ki,
    "chembl2034_ki": load_chembl2034_ki,
    "chembl2047_ec50": load_chembl2047_ec50,
    "chembl2147_ki": load_chembl2147_ki,
    "chembl2835_ki": load_chembl2835_ki,
    "chembl2971_ki": load_chembl2971_ki,
    "chembl3979_ec50": load_chembl3979_ec50,
    "chembl4005_ki": load_chembl4005_ki,
    "chembl4203_ki": load_chembl4203_ki,
    "chembl4616_ec50": load_chembl4616_ec50,
    "chembl4792_ki": load_chembl4792_ki,
}


@validate_params(
    {
        "subset": [None, list],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculeace_benchmark(
    subset: list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the MoleculeACE benchmark datasets.

    MoleculeACE [1]_ datasets are varied inhibition and effective concentration targets from ChEMBL [2]_.
    Activity cliffs split is recommended for all of them.

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    - chembl204_ki
    - chembl214_ki
    - chembl218_ec50
    - chembl219_ki
    - chembl228_ki
    - chembl231_ki
    - chembl233_ki
    - chembl234_ki
    - chembl235_ec50
    - chembl236_ki
    - chembl237_ec50
    - chembl237_ki
    - chembl238_ki
    - chembl239_ec50
    - chembl244_ki
    - chembl262_ki
    - chembl264_ki
    - chembl287_ki
    - chembl1862_ki
    - chembl1871_ki
    - chembl2034_ki
    - chembl2047_ec50
    - chembl2147_ki
    - chembl2835_ki
    - chembl2971_ki
    - chembl3979_ec50
    - chembl4005_ki
    - chembl4203_ki
    - chembl4616_ec50
    - chembl4792_ki

    Parameters
    ----------
    subset : None or list of strings
        If ``None``, returns all datasets. List of strings loads only datasets with given names.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    force_update : bool, default=False
        If True, always re-download the dataset from HuggingFace Hub, even if
        it is already present locally. If False, the dataset is downloaded only
        if it is not yet available locally.

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` parameter, either:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022.
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023.
        <https://doi.org/10.1093/nar/gkad1004>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
    ]

    if as_frames:
        datasets = (
            (
                dataset_name,
                load_function(
                    data_dir, as_frame=True, verbose=verbose, force_update=force_update
                ),
            )
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    else:
        datasets = (
            (
                dataset_name,
                *load_function(
                    data_dir, as_frame=False, verbose=verbose, force_update=force_update
                ),
            )
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    return datasets


@validate_params(
    {
        "dataset_name": [StrOptions(set(MOLECULEACE_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculeace_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load MoleculeACE dataset by name.

    Loads a given dataset from MoleculeACE [1]_ benchmark by its name. This is a proxy
    for easier benchmarking that avoids looking for individual functions.

    Dataset names here are the same as returned by :py:func:`.load_moleculenet_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and labels
        (dataset-dependent). Otherwise, returns SMILES as list of strings, and
        labels as a NumPy array (shape and type are dataset-dependent).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    force_update : bool, default=False
        If True, always re-download the dataset from HuggingFace Hub, even if
        it is already present locally. If False, the dataset is downloaded only
        if it is not yet available locally.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns depending on the dataset
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022.
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    Examples
    --------
    >> from skfp.datasets.moleculeace import load_moleculeace_dataset
    >> dataset = load_moleculeace_dataset("chembl204_ki")
    >> dataset   # doctest: +SKIP
    (['CCCCCCCC(=O)OC[C@H](NC(=O)CN)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O', ..., '])
    """
    loader_func = MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose, force_update)


@validate_params(
    {
        "dataset_name": [StrOptions(set(MOLECULEACE_DATASET_NAMES))],
        "split_type": [StrOptions({"random", "activity_cliff"})],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculeace_splits(
    dataset_name: str,
    split_type: str = "activity_cliff",
    data_dir: str | os.PathLike | None = None,
    as_dict: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> tuple[list[int], list[int]] | dict[str, list[int]]:
    """
    Load pre-generated dataset splits for the MoleculeACE benchmark.

    MoleculeACE [1]_ provides two stratified split types based on activity-cliff membership.
    The data are split into train/test partitions as one of:

    * ``random``
    * ``activity_cliff``

    Random splits use an 80/20 train/test split. Activity cliffs additionally
    restrict the test set to molecules that are part of activity-cliff pairs.
    Activity cliffs splits are recommended in the literature.

    Dataset names are the same as those returned by :py:func:`.load_moleculeace_benchmark`
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load splits for.

    split_type: {"random", "activity_cliff"}
        Type of the split to load.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train", "valid" and "test",
        and index lists as values. Otherwise, returns three lists with splits indexes.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    force_update : bool, default=False
        If True, always re-download the dataset from HuggingFace Hub, even if
        it is already present locally. If False, the dataset is downloaded only
        if it is not yet available locally.

    Returns
    -------
    data : tuple(list[int], list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - three lists of integer indexes
        - dictionary with "train", "valid" and "test" keys, and values as lists with
        splits indexes

    References
    ----------
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022.
        <https://doi.org/10.1021/acs.jcim.2c01073>`_
    """
    if split_type == "random":
        splits_suffix = "splits.json"
    elif split_type == "activity_cliff":
        splits_suffix = "splits_activity.json"
    else:
        raise ValueError(
            f'Split type "{split_type}" not recognized, must be one of: '
            f'{{"random", "activity_cliff"}}'
        )

    splits = fetch_splits(
        data_dir,
        dataset_name=f"MoleculeACE_{dataset_name}",
        filename=f"{dataset_name}_{splits_suffix}",
        verbose=verbose,
        force_update=force_update,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["test"]


def _subset_to_dataset_names(subset: list[str] | None) -> list[str]:
    if subset is None:
        dataset_names = MOLECULEACE_DATASET_NAMES
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in MOLECULEACE_DATASET_NAMES:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among MoleculeACE datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be a list of strings'
            f"with dataset names from MoleculeACE to load"
        )
    return dataset_names
