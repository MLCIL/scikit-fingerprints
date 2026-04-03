import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .expansionrx import (
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

EXPANSIONRX_DATASET_NAMES = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "RLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

EXPANSIONRX_DATASET_NAME_TO_LOADER_FUNC = {
    "LogD": load_logd,
    "KSOL": load_ksol,
    "HLM CLint": load_hlm_clint,
    "RLM CLint": load_rlm_clint,
    "MLM CLint": load_mlm_clint,
    "Caco-2 Permeability Papp A>B": load_caco2_perm_papp_a_b,
    "Caco-2 Permeability Efflux": load_caco2_perm_efflux,
    "MPPB": load_mppb,
    "MBPB": load_mbpb,
    "MGMB": load_mgmb,
}


@validate_params(
    {
        "subset": [None, list],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_expansionrx_benchmark(
    subset: list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the ExpansionRx-OpenADMET challenge datasets.

    Expansion Therapeutics - OpenADMET challenge [1]_ datasets come from real-work ADMET
    experiments from a series of drug discovery campaigns by Expansion Therapeutics on
    RNA-mediated diseases. This data has been obtained during late-stage optimization and
    has time-ordering information - IDs in the "Molecule name" column reflect measurement
    order.

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    - LogD
    - KSOL
    - HLM CLint
    - RLM CLint
    - MLM CLint
    - Caco-2 Permeability Papp A>B
    - Caco-2 Permeability Efflux
    - MPPB
    - MBPB
    - MGMB

    Note that RLM CLint has not been a part of the original challenge. It has been provided
    by the organizers afterward as an additional endpoint.

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

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` parameter, either:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `OpenADMET team
        "Announcement 1: ExpansionRx-OpenADMET Blind Challenge"
        <https://openadmet.ghost.io/expansionrx-openadmet-blind-challenge/>`_

    .. [2] `HuggingFace Hub - ExpansionRx-OpenADMET challenge full dataset
        <https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        EXPANSIONRX_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
    ]

    if as_frames:
        datasets = (
            (dataset_name, load_function(data_dir, as_frame=True, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    else:
        datasets = (
            (dataset_name, *load_function(data_dir, as_frame=False, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    return datasets


@validate_params(
    {
        "dataset_name": [StrOptions(set(EXPANSIONRX_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_expansionrx_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load ExpansionRx-OpenADMET challenge dataset by name.

    Loads a given dataset from ExpansionRx-OpenADMET challenge [1]_ by its name.
    This is a proxy for easier benchmarking that avoids looking for individual functions.

    Dataset names here are the same as returned by :py:func:`.load_expansionrx_benchmark` function,
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns depending on the dataset
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `OpenADMET team
        "Announcement 1: ExpansionRx-OpenADMET Blind Challenge"
        <https://openadmet.ghost.io/expansionrx-openadmet-blind-challenge/>`_

    Examples
    --------
    >> from skfp.datasets.expansionrx import load_expansionrx_dataset
    >> dataset = load_expansionrx_dataset("LogD")
    >> dataset   # doctest: +SKIP
    (['CN1CCC[C@H]1COc1ccc(-c2nc3cc(-c4ccc5[nH]c(-c6ccc(O)cc6)nc5c4)ccc3[nH]2)cc1', ..., '])
    """
    loader_func = EXPANSIONRX_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose)


@validate_params(
    {
        "dataset_name": [StrOptions(set(EXPANSIONRX_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_expansionrx_splits(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> tuple[list[int], list[int]] | dict[str, list[int]]:
    """
    Load pre-generated dataset splits for the ExpansionRx-OpenADMET challenge.

    ExpansionRx-OpenADMET challenge [1]_ provides time (chronological) split,
    based on the experiment order during late-stage ADMET optimization. 70/30
    train/test split is used, with no provided validation data. However, in
    Pandas DataFrame output, IDs in the "Molecule name" column are meaningful
    and indicate experiment order.

    Dataset names are the same as those returned by :py:func:`.load_expansionrx_benchmark`
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load splits for.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train", "valid" and "test",
        and index lists as values. Otherwise, returns three lists with splits indexes.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : tuple(list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - two lists of integer indexes
        - dictionary with "train" and "test" keys, and values as lists with
        splits indexes

    References
    ----------
    .. [1] `OpenADMET team
        "Announcement 1: ExpansionRx-OpenADMET Blind Challenge"
        <https://openadmet.ghost.io/expansionrx-openadmet-blind-challenge/>`_
    """
    splits = fetch_splits(
        data_dir,
        dataset_name=f"ExpansionRx_OpenADMET_{dataset_name}",
        filename=f"time_split_{dataset_name.lower()}.json",
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["test"]


def _subset_to_dataset_names(subset: list[str] | None) -> list[str]:
    if subset is None:
        dataset_names = EXPANSIONRX_DATASET_NAMES
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in EXPANSIONRX_DATASET_NAMES:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among "
                    f"ExpansionRx-OpenADMET challenge datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be a list of strings'
            f"with dataset names from ExpansionRx-OpenADMET challenge to load"
        )
    return dataset_names
