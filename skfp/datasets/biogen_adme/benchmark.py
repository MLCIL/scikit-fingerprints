import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from .biogen_adme import (
    load_hlm_clint,
    load_hppb,
    load_mdr1_mdck_er,
    load_rlm_clint,
    load_rppb,
    load_solubility,
)

BIOGEN_ADME_DATASET_NAMES = [
    "HLM CLint",
    "MDR1-MDCK ER",
    "Solubility",
    "hPPB",
    "rPPB",
    "RLM CLint",
]

BIOGEN_ADME_DATASET_NAME_TO_LOADER_FUNC = {
    "HLM CLint": load_hlm_clint,
    "MDR1-MDCK ER": load_mdr1_mdck_er,
    "Solubility": load_solubility,
    "hPPB": load_hppb,
    "rPPB": load_rppb,
    "RLM CLint": load_rlm_clint,
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
def load_biogen_adme_benchmark(
    subset: list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the Biogen ADME benchmark datasets.

    Biogen ADME benchmark [1]_ consists of 3521 diverse compounds from commercial
    libraries, tested against 6 in vitro ADME endpoints. All label values are
    log10-transformed. The "Internal ID" column reflects ordering and can be used
    for temporal splitting.

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    - HLM CLint
    - MDR1-MDCK ER
    - Solubility
    - hPPB
    - rPPB
    - RLM CLint

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
    .. [1] `Fang, Cheng, et al.
        "Prospective Validation of Machine Learning Algorithms for Absorption,
        Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective"
        J. Chem. Inf. Model. 2023, 63, 11, 3263-3274
        <https://doi.org/10.1021/acs.jcim.3c00160>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        BIOGEN_ADME_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
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
        "dataset_name": [StrOptions(set(BIOGEN_ADME_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_biogen_adme_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load Biogen ADME benchmark dataset by name.

    Loads a given dataset from Biogen ADME benchmark [1]_ by its name.
    This is a proxy for easier benchmarking that avoids looking for individual functions.

    Dataset names here are the same as returned by :py:func:`.load_biogen_adme_benchmark`
    function, and are case-sensitive.

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
    .. [1] `Fang, Cheng, et al.
        "Prospective Validation of Machine Learning Algorithms for Absorption,
        Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective"
        J. Chem. Inf. Model. 2023, 63, 11, 3263-3274
        <https://doi.org/10.1021/acs.jcim.3c00160>`_

    Examples
    --------
    >> from skfp.datasets.biogen_adme import load_biogen_adme_dataset
    >> dataset = load_biogen_adme_dataset("HLM CLint")
    >> dataset   # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., '])
    """
    loader_func = BIOGEN_ADME_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose)


def _subset_to_dataset_names(subset: list[str] | None) -> list[str]:
    if subset is None:
        dataset_names = BIOGEN_ADME_DATASET_NAMES
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in BIOGEN_ADME_DATASET_NAMES:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among "
                    f"Biogen ADME benchmark datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be a list of strings'
            f"with dataset names from Biogen ADME benchmark to load"
        )
    return dataset_names
