import os
from collections.abc import Iterator
from functools import partial

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.lrgb.peptides_func import load_peptides_func
from skfp.datasets.lrgb.peptides_struct import load_peptides_struct
from skfp.datasets.utils import fetch_splits


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "mol_type": [StrOptions({"SMILES", "aminoseq"})],
        "standardize_labels": ["boolean"],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_lrgb_mol_benchmark(
    data_dir: str | os.PathLike | None = None,
    mol_type: str = "SMILES",
    standardize_labels: bool = True,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the LRGB molecular datasets.

    There are two datasets: Peptides-func (binary multitask classification) and
    Peptides-struct (multitask regression). Stratified random split is recommended for
    both, following LRGB [1]_. See paper for details on stratification. AUPRC metric
    is recommended for Peptides-func, and MAE for Peptides-struct.

    Dataset names are also returned (case-sensitive): "Peptides-func", "Peptides-struct".

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    mol_type : {"SMILES", "aminoseq"}, default="SMILES"
        Which molecule representation to return, either SMILES strings or aminoacid
        sequences.

    standardize_labels : bool, default=True
        Whether to standardize labels to mean 0 and standard deviation 1 for
        Peptides-struct, following the recommendation from the original paper [1]_.
        Otherwise, the raw property values are returned.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` and ``mol_type`` parameters, either:
        - Pandas DataFrame with columns: "SMILES"/"aminoseq", "label"
        - tuple of: list of strings (SMILES / aminoacid sequences), NumPy array (labels)

    References
    ----------
    .. [1] `Dwivedi, Vijay Prakash, et al.
        "Long Range Graph Benchmark"
        Advances in Neural Information Processing Systems 35 (2022): 22326-22340
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html>`_
    """
    datasets = [
        (
            "Peptides-func",
            load_peptides_func,
        ),
        (
            "Peptides-struct",
            partial(load_peptides_struct, standardize_labels=standardize_labels),
        ),
    ]
    load_args = {
        "data_dir": data_dir,
        "mol_type": mol_type,
        "as_frame": as_frames,
        "verbose": verbose,
    }

    if as_frames:
        # generator of tuples (dataset_name, DataFrame)
        datasets_gen = (
            (dataset_name, load_function(**load_args))
            for dataset_name, load_function in datasets
        )
    else:
        # generator of tuples (dataset_name, SMILES/aminoseq, y)
        datasets_gen = (
            (dataset_name, *load_function(**load_args))
            for dataset_name, load_function in datasets
        )

    return datasets_gen


@validate_params(
    {
        "dataset_name": [StrOptions({"Peptides-func", "Peptides-struct"})],
        "data_dir": [None, str, os.PathLike],
        "mol_type": [StrOptions({"SMILES", "aminoseq"})],
        "standardize_labels": ["boolean"],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_lrgb_mol_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    mol_type: str = "SMILES",
    standardize_labels: bool = True,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load LRGB molecular dataset by name.

    Loads a given dataset from LRGB benchmark by its name. This is a proxy for
    easier benchmarking, that avoids looking for individual functions.

    Dataset names are (case-sensitive): "Peptides-func", "Peptides-struct".

    Parameters
    ----------
    dataset_name : {"Peptides-func", "Peptides-struct"}
        Name of the dataset to load.

    mol_type : {"SMILES", "aminoseq"}, default="SMILES"
        Which molecule representation to return, either SMILES strings or aminoacid
        sequences.

    standardize_labels : bool, default=True
        Whether to standardize labels to mean 0 and standard deviation 1 for
        Peptides-struct, following the recommendation from the original paper [1]_.
        Otherwise, the raw property values are returned.

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
    .. [1] `Dwivedi, Vijay Prakash, et al.
        "Long Range Graph Benchmark"
        Advances in Neural Information Processing Systems 35 (2022): 22326-22340
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html>`_

    Examples
    --------
    >>> from skfp.datasets.lrgb import load_lrgb_mol_dataset
    >>> dataset = load_lrgb_mol_dataset("Peptides-func")
    >>> dataset  # doctest: +SKIP
    (['[Cl].CC(C)NCC(O)COc1cccc2ccccc12', ..., '[N+](=NCC(=O)N[C@@H]([C@H](O)C1=CC=C([N+]([O-])=O)C=C1)CO)=[N-]'], \
array([1, 1, 1, ..., 1, 1, 1]))

    >>> dataset = load_lrgb_mol_dataset("Peptides-func", as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES  label
    0                   [Cl].CC(C)NCC(O)COc1cccc2ccccc12      1
    1           C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl      1
    2  c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO...      1
    3                   C1CCN(CC1)Cc1cccc(c1)OCCCNC(=O)C      1
    4  Cc1onc(c2ccccc2Cl)c1C(=O)N[C@H]3[C@H]4SC(C)(C)...      1
    """
    if dataset_name == "Peptides-func":
        return load_peptides_func(data_dir, mol_type, as_frame, verbose)
    else:
        return load_peptides_struct(
            data_dir, mol_type, standardize_labels, as_frame, verbose
        )


@validate_params(
    {
        "dataset_name": [StrOptions({"Peptides-func", "Peptides-struct"})],
        "valid_sequences_only": ["boolean"],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_lrgb_mol_splits(
    dataset_name: str,
    valid_sequences_only: bool = False,
    data_dir: str | os.PathLike | None = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> tuple[list[int], list[int], list[int]] | dict[str, list[int]]:
    """
    Load the official LRGB splits for molecular datasets.

    Long Range Graph Benchmark (LRGB) [1]_ uses precomputed stratified random split for
    both Peptides-func and Peptides-struct datasets.

    Dataset names here are the same as returned by `load_moleculenet_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : {"Peptides-func", "Peptides-struct"}
        Name of the dataset to load splits for.

    valid_sequences_only : bool, default=False
        Whether to load only rows with valid amino acid sequences, which can be loaded
        as RDKit ``Mol`` objects. This removes some sequences with valid SMILES, but custom
        notation for chemical modifications.

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
    data : tuple(list[int], list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - three lists of integer indexes
        - dictionary with "train", "valid" and "test" keys, and values as lists with
        splits indexes

    References
    ----------
    .. [1] `Dwivedi, Vijay Prakash, et al.
        "Long Range Graph Benchmark"
        Advances in Neural Information Processing Systems 35 (2022): 22326-22340
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html>`_
    """
    file_dataset_name = dataset_name.lower().replace("-", "_")
    valid_only = "_valid_seqs" if valid_sequences_only else ""

    splits = fetch_splits(
        data_dir,
        dataset_name=f"LRGB_{dataset_name}",
        filename=f"lrgb_splits_{file_dataset_name}{valid_only}.json",
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["valid"], splits["test"]
