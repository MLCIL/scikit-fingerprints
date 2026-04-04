import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .asap import (
    load_hlm,
    load_ksol,
    load_logd,
    load_mdr1_mdckii,
    load_mlm,
    load_pic50_mers_cov,
    load_pic50_sars_cov_2,
)

ASAP_DATASET_NAMES = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
    "pIC50 SARS-CoV-2",
    "pIC50 MERS-CoV",
]

ASAP_DATASET_NAME_TO_LOADER_FUNC = {
    "HLM": load_hlm,
    "KSOL": load_ksol,
    "LogD": load_logd,
    "MDR1-MDCKII": load_mdr1_mdckii,
    "MLM": load_mlm,
    "pIC50 SARS-CoV-2": load_pic50_sars_cov_2,
    "pIC50 MERS-CoV": load_pic50_mers_cov,
}

_ASAP_DATASET_NAME_TO_HF_NAME = {
    "HLM": "ASAP_OpenADMET_HLM",
    "KSOL": "ASAP_OpenADMET_KSOL",
    "LogD": "ASAP_OpenADMET_LogD",
    "MDR1-MDCKII": "ASAP_OpenADMET_MDR1-MDCKII",
    "MLM": "ASAP_OpenADMET_MLM",
    "pIC50 SARS-CoV-2": "ASAP_OpenADMET_pIC50_SARS-CoV-2",
    "pIC50 MERS-CoV": "ASAP_OpenADMET_pIC50_MERS-CoV",
}

_ASAP_DATASET_NAME_TO_SPLIT_FILENAME = {
    "HLM": "time_split_hlm.json",
    "KSOL": "time_split_ksol.json",
    "LogD": "time_split_logd.json",
    "MDR1-MDCKII": "time_split_mdr1_mdckii.json",
    "MLM": "time_split_mlm.json",
    "pIC50 SARS-CoV-2": "time_split_pic50_sars_cov_2.json",
    "pIC50 MERS-CoV": "time_split_pic50_mers_cov.json",
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
def load_asap_benchmark(
    subset: list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the ASAP Discovery-Polaris-OpenADMET challenge datasets.

    ASAP Discovery - Polaris - OpenADMET challenge [1]_ [2]_ [3]_ datasets come from antiviral
    drug discovery campaigns by the ASAP Discovery consortium, targeting SARS-CoV-2 and
    MERS-CoV main protease (Mpro) inhibitors. The challenge included ADMET and potency
    endpoints.

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    - HLM
    - KSOL
    - LogD
    - MDR1-MDCKII
    - MLM
    - pIC50 SARS-CoV-2
    - pIC50 MERS-CoV

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
    .. [1] `ASAP Discovery
        "ASAP Discovery x OpenADMET Antiviral Drug Discovery Challenge"
        <https://polarishub.io/blog/antiviral-competition>`_

    .. [2] `Chodera et al.
        "The ASAP Discovery Antiviral Drug Discovery Challenge"
        <https://doi.org/10.26434/chemrxiv-2025-zd9mr>`_

    .. [3] `MacDermott-Opeskin, Hugo, et al.
        "A computational community blind challenge on pan-coronavirus drug discovery data"
        J. Chem. Inf. Model. 2026, 66, 6, 3129-3149
        <https://doi.org/10.1021/acs.jcim.5c02106>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        ASAP_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
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
        "dataset_name": [StrOptions(set(ASAP_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_asap_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load ASAP Discovery-OpenADMET challenge dataset by name.

    Loads a given dataset from ASAP Discovery-OpenADMET challenge [1]_ by its name.
    This is a proxy for easier benchmarking that avoids looking for individual functions.

    Dataset names here are the same as returned by :py:func:`.load_asap_benchmark` function,
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
    .. [1] `ASAP Discovery
        "ASAP Discovery x OpenADMET Antiviral Drug Discovery Challenge"
        <https://polarishub.io/blog/antiviral-competition>`_

    Examples
    --------
    >> from skfp.datasets.asap import load_asap_dataset
    >> dataset = load_asap_dataset("LogD")
    >> dataset   # doctest: +SKIP
    (['COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1', ..., '])
    """
    loader_func = ASAP_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose)


@validate_params(
    {
        "dataset_name": [StrOptions(set(ASAP_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_asap_splits(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> tuple[list[int], list[int]] | dict[str, list[int]]:
    """
    Load pre-generated dataset splits for the ASAP Discovery-OpenADMET challenge.

    ASAP Discovery-OpenADMET challenge [1]_ provides time (chronological) split,
    based on the competition train/test partition. No validation data is provided.

    Dataset names are the same as those returned by :py:func:`.load_asap_benchmark`
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load splits for.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train" and "test",
        and index lists as values. Otherwise, returns two lists with splits indexes.

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
    .. [1] `ASAP Discovery
        "ASAP Discovery x OpenADMET Antiviral Drug Discovery Challenge"
        <https://polarishub.io/blog/antiviral-competition>`_
    """
    splits = fetch_splits(
        data_dir,
        dataset_name=_ASAP_DATASET_NAME_TO_HF_NAME[dataset_name],
        filename=_ASAP_DATASET_NAME_TO_SPLIT_FILENAME[dataset_name],
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["test"]


def _subset_to_dataset_names(subset: list[str] | None) -> list[str]:
    if subset is None:
        dataset_names = ASAP_DATASET_NAMES
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in ASAP_DATASET_NAMES:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among "
                    f"ASAP Discovery-OpenADMET challenge datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be a list of strings'
            f"with dataset names from ASAP Discovery-OpenADMET challenge to load"
        )
    return dataset_names
