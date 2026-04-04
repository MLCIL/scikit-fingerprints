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
def load_hlm(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the HLM dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the human liver microsomal stability (HLM), i.e.
    intrinsic clearance in uL/min/mg, of antiviral compounds targeting
    SARS-CoV-2 and MERS-CoV main protease (Mpro) [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   407
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_hlm
    >>> dataset = load_hlm()
    >>> dataset  # doctest: +SKIP
    (['CN(C1=CC=C2CNCC2=C1)[C@H](C(=O)NCC(F)F)C1=CC(Cl)=CC(C2CC2)=C1', ..., 'CC1=NC=C(CN2C[C@@]3(C(=O)N(C4=CN=CC5=CC=CC=C45)C[C@@H]3C)C3=CC(Cl)=CC=C32)C(=O)N1'], \
    array([17.1, ..., 143.]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_HLM",
        filename="hlm.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_ksol(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the KSOL dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the kinetic solubility (KSOL) in uM of antiviral
    compounds targeting SARS-CoV-2 and MERS-CoV main protease (Mpro) [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   477
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_ksol
    >>> dataset = load_ksol()
    >>> dataset  # doctest: +SKIP
    (['O=C(NCC(F)F)[C@H](NC1=CC2=C(C=C1Br)CNC2)C1=CC(Cl)=CC(C2CC2)=C1', ..., 'COC[C@H]1CN(C2=CN=CC3=CC=CC=C23)C(=O)[C@@]12CN(CC1=CC=NN1)C(=O)C2'], \
    array([333., ..., 397.]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_KSOL",
        filename="ksol.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_logd(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the LogD dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the distribution coefficient (LogD) of antiviral
    compounds targeting SARS-CoV-2 and MERS-CoV main protease (Mpro) [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   478
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_logd
    >>> dataset = load_logd()
    >>> dataset  # doctest: +SKIP
    (['COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1', ..., 'CC1=NC=C(CN2C[C@@]3(C(=O)N(C4=CN=CC5=CC=CC=C45)C[C@@H]3C)C3=CC(Cl)=CC=C32)C(=O)N1'], \
    array([0.3, ..., 2. ]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_LogD",
        filename="logd.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_mdr1_mdckii(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the MDR1-MDCKII dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the permeability through MDR1-expressing MDCK-II cells
    (in 10^-6 cm/s) of antiviral compounds targeting SARS-CoV-2 and MERS-CoV main
    protease (Mpro) [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   551
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_mdr1_mdckii
    >>> dataset = load_mdr1_mdckii()
    >>> dataset  # doctest: +SKIP
    (['COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1', ..., 'CC1=NC=C(CN2C[C@@]3(C(=O)N(C4=CN=CC5=CC=CC=C45)C[C@@H]3C)C3=CC(Cl)=CC=C32)C(=O)N1'], \
    array([2. , ..., 5.1]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_MDR1-MDCKII",
        filename="mdr1_mdckii.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_mlm(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the MLM dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the mouse liver microsomal stability (MLM), i.e.
    intrinsic clearance in uL/min/mg, of antiviral compounds targeting
    SARS-CoV-2 and MERS-CoV main protease (Mpro) [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   425
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_mlm
    >>> dataset = load_mlm()
    >>> dataset  # doctest: +SKIP
    (['CC(C)NC(=O)[C@H](NC1=CC=C2CNCC2=C1)C1=CC(Cl)=CC2=C1N=C(C1=CC=CC=C1)N2', ..., 'CC1=NC=C(CN2C[C@@]3(C(=O)N(C4=CN=CC5=CC=CC=C45)C[C@@H]3C)C3=CC(Cl)=CC=C32)C(=O)N1'], \
    array([ 11., ..., 259.]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_MLM",
        filename="mlm.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_pic50_sars_cov_2(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the pIC50 SARS-CoV-2 dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the pIC50 (negative log10 of IC50 in uM) of antiviral
    compounds against SARS-CoV-2, measured by fluorescence dose-response
    assay [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1105
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_pic50_sars_cov_2
    >>> dataset = load_pic50_sars_cov_2()
    >>> dataset  # doctest: +SKIP
    (['C=C(CN1CCC2=C(C=C(Cl)C=C2)C1C(=O)NC1=CN=CC2=CC=CC=C12)C(N)=O', ..., 'COC1=CC=CC=C1[C@H]1C[C@H](C)CCN1C(=O)CC1=CN=CC2=CC=CC=C12'], \
    array([5.29, ..., 5.77]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_pIC50_SARS-CoV-2",
        filename="pic50_sars_cov_2.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_pic50_mers_cov(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the pIC50 MERS-CoV dataset from the ASAP Discovery-OpenADMET challenge.

    The task is to predict the pIC50 (negative log10 of IC50 in uM) of antiviral
    compounds against MERS-CoV, measured by fluorescence dose-response
    assay [1]_ [2]_ [3]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1198
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

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
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

    Examples
    --------
    >>> from skfp.datasets.asap import load_pic50_mers_cov
    >>> dataset = load_pic50_mers_cov()
    >>> dataset  # doctest: +SKIP
    (['COC[C@]1(C)C(=O)N(C2=CN=CC3=CC=CC=C23)C(=O)N1C', ..., 'COC1=CC=CC=C1[C@H]1C[C@H](C)CCN1C(=O)CC1=CN=CC2=CC=CC=C12'], \
    array([4.19, ..., 5.47]))
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="ASAP_OpenADMET_pIC50_MERS-CoV",
        filename="pic50_mers_cov.csv",
        verbose=verbose,
    )
    return (
        df
        if as_frame
        else get_mol_strings_and_labels(df, non_target_columns="Molecule name")
    )
