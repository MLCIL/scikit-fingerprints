import os

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import validate_params

from skfp.datasets.utils import fetch_dataset, get_mol_strings_and_labels


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_sarscov2_3clpro_diamond(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the SARS-CoV-2 3CL Protease Diamond dataset.

    XChem crystallographic fragment screen against SARS-CoV-2 main protease at high resolution [1]_ [2]_.
    The task is to predict molecules' activity against SARS-CoV-2 3CL Protease.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      880
    Recommended split             scaffold
    Recommended metric               AUROC
    ==================   =================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D integer binary
        vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `"Diamond Coronavirus Science - Main protease structure and XChem fragment screen"
        <https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem.html>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_sarscov2_3clpro_diamond",
        filename="tdc_sarscov2_3clpro_diamond.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_sarscov2_vitro_touret(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the SARSCoV2 Vitro Touret dataset.

    An in-vitro screen of the Prestwick chemical library of drugs in an infected cell-based assay [1]_ [2]_.
    The task is to predict molecules' activity against SARSCoV2.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     1484
    Recommended split             scaffold
    Recommended metric               AUROC
    ==================   =================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D integer binary
        vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] Touret, F., Gilles, M., Barral, K. et al.
        "In vitro screening of a FDA approved chemical library reveals potential inhibitors of SARS-CoV-2 replication"
        Sci Rep 10, 13093 (2020).
        <https://doi.org/10.1038/s41598-020-70143-6>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_sarscov2_vitro_touret",
        filename="tdc_sarscov2_vitro_touret.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
