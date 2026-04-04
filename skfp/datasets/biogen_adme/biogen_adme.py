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
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_hlm_clint(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the HLM CLint dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of human liver microsomal intrinsic
    clearance (HLM CLint, in mL/min/kg) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3087
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_hlm_clint
    >>> dataset = load_hlm_clint()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'Cc1cccc(/C=N/Nc2cc(N3CCOCC3)n3nc(-c4ccncc4)cc3n2)c1'], \
    array([0.676, ..., 1.507]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_HLM_CLint",
        filename="hlm_clint.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_mdr1_mdck_er(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the MDR1-MDCK ER dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of MDR1-MDCK efflux ratio
    (B-A/A-B) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2642
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_mdr1_mdck_er
    >>> dataset = load_mdr1_mdck_er()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'O=C(Nc1nc2ccccc2[nH]1)c1ccc(-n2cccc2)cc1'], \
    array([ 1.493, ..., -0.444]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_MDR1-MDCK_ER",
        filename="mdr1_mdck_er.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_solubility(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the Solubility dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of aqueous solubility at pH 6.8
    (in ug/mL) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2173
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_solubility
    >>> dataset = load_solubility()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'CN1CC(=O)N(CCOc2ccccc2)C1=O'], \
    array([0.09 , ..., 1.363]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_Solubility",
        filename="solubility.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_hppb(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the hPPB dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of human plasma protein binding
    (% unbound) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   194
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_hppb
    >>> dataset = load_hppb()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'c1sc(NCC2CCCO2)nc1C12CC3CC(CC(C3)C1)C2'], \
    array([ 0.991, ..., -1.097]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_hPPB",
        filename="hppb.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_rppb(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the rPPB dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of rat plasma protein binding
    (% unbound) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   168
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_rppb
    >>> dataset = load_rppb()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'CC(C)[C@@](C)(O)[C@@H]1CN(c2nc(-c3n[nH]c4ncccc34)c(F)cc2Cl)CCN1'], \
    array([0.519, ..., 0.622]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_rPPB",
        filename="rppb.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
        "force_update": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_rlm_clint(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
    force_update: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the RLM CLint dataset from the Biogen ADME benchmark.

    The task is to predict the log10 of rat liver microsomal intrinsic
    clearance (RLM CLint, in mL/min/kg) of molecules [1]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3054
    Recommended split              time
    Recommended metric              MAE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

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
        - Pandas DataFrame with columns: "SMILES", "label"
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
    >>> from skfp.datasets.biogen_adme import load_rlm_clint
    >>> dataset = load_rlm_clint()
    >>> dataset  # doctest: +SKIP
    (['CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12', ..., 'C[C@@H](CN1CCC(n2c(=O)[nH]c3cc(Br)ccc32)CC1)NC(=O)[C@@H]1C[C@H]1c1ccccc1'], \
    array([1.392, ..., 2.759]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="Biogen_ADME_RLM_CLint",
        filename="rlm_clint.csv",
        verbose=verbose,
        force_update=force_update,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Internal ID")
    )
