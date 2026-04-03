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
def load_chembl204_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL204 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Prothrombin target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2754
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl204_ki
    >>> dataset = load_chembl204_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(=N)N)cc5c4)c3c2)CC1, ..., 'CCC(=O)N1CCC[C@H]1C(=O)NCc1ccc(C(=N)N)cc1'], \
    array([-3.427, ..., -4.146]))

    >>> dataset = load_chembl204_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(... -3.426511
    1  CC(=N)N1CCC(Oc2ccc3c(c2)nc(C(C)C)n3Cc2ccc3ccc(... -2.939519
    2  CCC(C)c1nc2cc(OC3CCN(C(C)=N)CC3)ccc2n1Cc1ccc2c... -3.361728
    3  COC(=O)C(C)CN(c1ccc2c(c1)nc(C)n2Cc1ccc2ccc(C(=... -3.698970
    4  CCCCc1nc2cc(OC3CCN(C(C)=N)CC3)ccc2n1Cc1ccc2ccc... -3.301030
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl204_ki",
        filename="chembl204_ki.csv",
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
def load_chembl214_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL214 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the 5-hydroxytryptamine receptor 1a target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3317
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl214_ki
    >>> dataset = load_chembl214_ki()
    >>> dataset  # doctest: +SKIP
    (['COc1ccc(NC(=O)c2ccc(-c3ccc(-c4noc(C)n4)cc3C)cc2)cc1N1CCN(C)CC1, ..., 'O=S(=O)(NCCCCCCN1CCN(c2nsc3ccccc23)CC1)c1ccc2ccccc2c1'], \
    array([-1.869, ..., -1.863]))

    >>> dataset = load_chembl214_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  COc1ccc(NC(=O)c2ccc(-c3ccc(-c4noc(C)n4)cc3C)cc... -1.869232
    1   Nc1cccc(-c2ccc(CCN3CCN(c4cccc5cccnc45)CC3)cc2)n1 -0.477121
    2      COc1ccc(NS(=O)(=O)c2ccc(Br)cc2)cc1N1CCN(C)CC1 -2.400002
    3  COc1ccc(NS(=O)(=O)c2sc3ccc(Cl)cc3c2C)cc1N1CCN(... -2.700002
    4         CN1CCc2cccc3c2[C@H]1Cc1cccc(-c2ccccc2)c1-3 -0.255273

    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl214_ki",
        filename="chembl214_ki.csv",
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
def load_chembl218_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL218 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Cannabinoid receptor 1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1031
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl218_ec50
    >>> dataset = load_chembl218_ec50()
    >>> dataset  # doctest: +SKIP
    (['Cn1c(C(=O)NN2CCCCC2)nc(-c2ccc(Cl)cc2)c1-c1ccc(Cl)cc1, ..., 'CCCCCc1cccc(OCCCCCCCCCCC(=O)NC2CC2)c1'], \
    array([-2.0, ..., -1.491]))

    >>> dataset = load_chembl218_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  Cn1c(C(=O)NN2CCCCC2)nc(-c2ccc(Cl)cc2)c1-c1ccc(... -2.000000
    1  Cn1c(C(=O)NC2CCCCC2)nc(-c2ccc(Cl)cc2)c1-c1ccc(... -2.698970
    2  Cn1c(C(=O)NN2CCCCC2)nc(-c2ccc(Cl)cc2Cl)c1-c1cc... -0.698970
    3  Cn1c(C(=O)NC2CCCCC2)nc(-c2ccc(Cl)cc2Cl)c1-c1cc... -1.255273
    4  N#Cc1cc(-c2ccc(Cl)cc2)c(-c2ccc(Cl)cc2Cl)nc1OCc... -0.903090
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl218_ec50",
        filename="chembl218_ec50.csv",
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
def load_chembl219_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL219 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the D(4) dopamine receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1865
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl219_ki
    >>> dataset = load_chembl219_ki()
    >>> dataset  # doctest: +SKIP
    (['COc1ccccc1N1CCN(Cc2ccn(-c3ccccc3)c2)CC1, ..., 'CNc1cc(OC)c(C(=O)N[C@@H]2CCN(Cc3ccccc3)[C@@H]2C)cc1Cl'], \
    array([-0.1139, ..., 0.0655]))

    >>> dataset = load_chembl219_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0            COc1ccccc1N1CCN(Cc2ccn(-c3ccccc3)c2)CC1 -0.113943
    1            c1ccc(N2CCN(Cc3ccn(-c4ccccc4)c3)CC2)cc1 -0.602060
    2  CC1Cc2cccc3c2N1C(=O)C(N1CCN(Cc2ccc(Cl)cc2)CC1)CC3 -0.954243
    3  CC1(C)Cc2cccc3c2N1C(=O)C(N1CCN(Cc2ccc(Cl)cc2)C... -1.278754
    4      Cc1ccc(CN2CCN(C3CCc4cccc5c4N(CC5)C3=O)CC2)cc1 -0.602060
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl219_ki",
        filename="chembl219_ki.csv",
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
def load_chembl228_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL228 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Sodium-dependent serotonin transporter target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1704
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl228_ki
    >>> dataset = load_chembl228_ki()
    >>> dataset  # doctest: +SKIP
    (['CN(C)Cc1ccccc1Sc1ccc(C#N)cc1N, ..., 'CCCN(CC[C@]1(O)C[C@H](NC(=O)c2ccc3ccccc3c2)C1)[C@H]1CCc2nc(N)sc2C1'], \
    array([-0.04139, ..., -1.505]))

    >>> dataset = load_chembl228_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                              SMILES        Ki
    0                  CN(C)Cc1ccccc1Sc1ccc(C#N)cc1N -0.041393
    1             CN(C)Cc1ccccc1Sc1ccc(C(F)(F)F)cc1N  0.481486
    2                 COc1ccc(Sc2ccccc2CN(C)C)c(N)c1 -0.276462
    3                   CN(C)Cc1ccccc1Sc1ccc(Cl)cc1N  0.568636
    4  Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3c(c2)OCO3)cc1  0.661986
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl228_ki",
        filename="chembl228_ki.csv",
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
def load_chembl231_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL231 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Histamine h1 receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   973
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl231_ki
    >>> dataset = load_chembl231_ki()
    >>> dataset  # doctest: +SKIP
    (['CN1CCN(C2=Nc3ccccc3Nc3sc(CO)cc32)CC1, ..., 'O=C(O)c1cc(-c2ccc(C3CCNCC3)cc2)cc(-n2cc(-c3ccc(Cl)s3)nn2)c1'], \
    array([-0.7782, ..., -2.23]))

    >>> dataset = load_chembl231_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0               CN1CCN(C2=Nc3ccccc3Nc3sc(CO)cc32)CC1 -0.778151
    1                Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1 -0.622900
    2                   Cc1cc2c(s1)Nc1ccccc1N=C2N1CCNCC1 -1.342423
    3       Cc1cc2c(s1)Nc1ccccc1N=C2N1CC[N+](C)([O-])CC1 -1.939519
    4  CC(=O)c1ccc(OCCCN2CC[C@H](NC(=O)[C@@H](N)CO)C2... -4.633468
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl231_ki",
        filename="chembl231_ki.csv",
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
def load_chembl233_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL233 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Mu-type opioid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3142
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl233_ki
    >>> dataset = load_chembl233_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1, ..., 'CCO[C@@]12Cc3cc(-c4ccccc4)cnc3[C@@H]3Oc4c(O)ccc5c4[C@@]31CCN(CC1CC1)[C@@H]2C5'], \
    array([-4.026, ..., -2.698]))

    >>> dataset = load_chembl233_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0        CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1 -4.026125
    1  Cc1ccc(C(c2ccc(C)cc2)N2CC[C@H]2[C@H](N)c2cccc(... -2.903633
    2  COc1ccc([C@H](N)[C@@H]2CCN2C(c2ccccc2)c2ccccc2... -2.937016
    3  N[C@H](c1cccc(Cl)c1)[C@@H]1CCN1C(c1ccc(F)cc1)c... -3.337659
    4  N[C@H](c1cccc(Cl)c1)[C@@H]1CCN1C(c1cccc(Cl)c1)... -3.854852
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl233_ki",
        filename="chembl233_ki.csv",
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
def load_chembl234_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL234 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the D(3) dopamine receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3657
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl234_ki
    >>> dataset = load_chembl234_ki()
    >>> dataset  # doctest: +SKIP
    (['CN1C2CCC1CC(OC(c1ccc(F)cc1)c1ccc(F)cc1)C2, ..., 'CNc1cc(OC)c(C(=O)N[C@@H]2CCN(Cc3ccccc3)[C@@H]2C)cc1Cl'], \
    array([-2.161, ..., -0.07188]))

    >>> dataset = load_chembl234_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0          CN1C2CCC1CC(OC(c1ccc(F)cc1)c1ccc(F)cc1)C2 -2.161368
    1  O=C(NCCCN1CCN(c2cccc(Cl)c2Cl)CC1)c1cccc2c1-c1c... -1.556303
    2         c1ccc(N2CCN(CCCn3c4ccccc4c4ccccc43)CC2)cc1 -3.383815
    3             Oc1nc2c(N3CCN(Cc4ccccc4)CC3)cccc2[nH]1 -1.752048
    4  O=C(NCCCN1CCN(c2ccccc2)CC1)c1cccc2c1-c1ccccc1C2=O -2.633468
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl234_ki",
        filename="chembl234_ki.csv",
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
def load_chembl235_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL235 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Peroxisome proliferator-activated receptor gamma target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2349
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl235_ec50
    >>> dataset = load_chembl235_ec50()
    >>> dataset  # doctest: +SKIP
    (['CC(/C=C/C(F)=C(/C)c1cc(C(C)(C)C)cc(C(C)(C)C)c1OCC(F)(F)F)=C\C(=O)O, ..., 'O=C(O)Cc1cc(Br)c(Oc2cc(I)c(O)c(I)c2)c(I)c1'], \
    array([-1.324, ..., -2.477]))

    >>> dataset = load_chembl235_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  CC(/C=C/C(F)=C(/C)c1cc(C(C)(C)C)cc(C(C)(C)C)c1... -1.324282
    1  CCCOc1c(/C(C)=C\C=C\C(C)=C\C(=O)O)cc(C(C)C)cc1... -1.343409
    2  C/C(=C/C=C/C(C)=C/C(=O)O)c1cc(-c2cccs2)cc(C(C)... -0.993436
    3  CCC(Cc1ccc(OC)c(C(=O)NCc2ccc(OCCc3ccccc3)cc2)c... -3.477121
    4  CCCCC(Cc1ccc(OC)c(C(=O)NCc2ccc(C(F)(F)F)cc2)c1... -3.397940
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl235_ec50",
        filename="chembl235_ec50.csv",
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
def load_chembl236_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL236 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Delta-type opioid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2598
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl236_ki
    >>> dataset = load_chembl236_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1, ..., 'CCO[C@@]12Cc3cc(-c4ccccc4)cnc3[C@@H]3Oc4c(O)ccc5c4[C@@]31CCN(CC1CC1)[C@@H]2C5'], \
    array([-4.592, ..., -0.8739]))

    >>> dataset = load_chembl236_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0        CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1 -4.592399
    1  O=C(CCC(=O)NCC(=O)N[C@@H]1CC[C@@]2(O)[C@H]3Cc4... -0.892095
    2  Cc1ccc(C(c2ccc(C)cc2)N2CC[C@H]2[C@H](N)c2cccc(... -3.699924
    3  O=C(CCC(=O)NCC(=O)NCC(=O)NCC(=O)N[C@@H]1CC[C@@... -0.806180
    4  COc1ccc([C@H](N)[C@@H]2CCN2C(c2ccccc2)c2ccccc2... -3.465234
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl236_ki",
        filename="chembl236_ki.csv",
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
def load_chembl237_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL237 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Kappa-type opioid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   955
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl237_ec50
    >>> dataset = load_chembl237_ec50()
    >>> dataset  # doctest: +SKIP
    (['C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@@]3(O)[C@H]1C5, ..., 'Oc1ccc2c3c1O[C@H]1c4ncc(-c5ccccc5)cc4C[C@@]4(OCCCC5CCCCC5)[C@@H](C2)N(CC2CC2)CC[C@]314'], \
    array([-0.9191, ..., -1.538]))

    >>> dataset = load_chembl237_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  CC[C@H](C)[C@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[...  0.130768
    1  C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@@]3... -0.919078
    2  CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@]24CC[C@@]3(... -1.320146
    3  CO[C@@]12CCC3(C[C@H]1[C@@](C)(O)C(C)(C)C)[C@H]... -0.380211
    4  Nc1nc2cc3c(cc2s1)C[C@@H]1[C@@H]2CCCC[C@]32CCN1... -0.380211
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl237_ec50",
        filename="chembl237_ec50.csv",
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
def load_chembl237_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL237 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Kappa-type opioid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2603
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl237_ki
    >>> dataset = load_chembl237_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1, ..., 'CCO[C@@]12Cc3cc(-c4ccccc4)cnc3[C@@H]3Oc4c(O)ccc5c4[C@@]31CCN(CC1CC1)[C@@H]2C5'], \
    array([-3.613, ..., -2.401]))

    >>> dataset = load_chembl237_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0        CC(c1ccccc1)N1CC[C@H]1[C@@H](N)c1cccc(Cl)c1 -3.612678
    1  Cc1ccc(C(c2ccc(C)cc2)N2CC[C@H]2[C@H](N)c2cccc(... -3.265054
    2  COc1ccc([C@H](N)[C@@H]2CCN2C(c2ccccc2)c2ccccc2... -3.127429
    3  N[C@H](c1cccc(Cl)c1)[C@@H]1CCN1C(c1ccc(F)cc1)c... -3.350248
    4  N[C@H](c1cccc(Cl)c1)[C@@H]1CCN1C(c1cccc(Cl)c1)... -3.780821
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl237_ki",
        filename="chembl237_ki.csv",
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
def load_chembl238_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL238 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Sodium-dependent dopamine transporter target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1052
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl238_ki
    >>> dataset = load_chembl238_ki()
    >>> dataset  # doctest: +SKIP
    (['CN1CCC(O)(c2ccc(Cl)c(Cl)c2)C([C@@H](O)c2ccc(Cl)c(Cl)c2)C1, ..., 'C[C@H]1CN(CC[S+](O)C(c2ccc(F)cc2)c2ccc(F)cc2)C[C@@H](C)N1CC(O)Cc1ccccc1'], \
    array([-3.617, ..., -0.873]))

    >>> dataset = load_chembl238_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CN1CCC(O)(c2ccc(Cl)c(Cl)c2)C([C@@H](O)c2ccc(Cl... -3.617000
    1  CN1CCC(O)(c2ccc(Cl)c(Cl)c2)C(C(=O)c2ccc(Cl)c(C... -1.037426
    2      Cc1ccc(C2OC(=O)OC3(c4ccc(C)cc4)CCN(C)CC23)cc1 -3.913284
    3       Cc1ccc([C@H](O)C2CN(C)CCC2(O)c2ccc(C)cc2)cc1 -4.027350
    4        CN1CCC(O)(c2ccc(F)cc2)C(C(=O)c2ccc(F)cc2)C1 -3.755875
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl238_ki",
        filename="chembl238_ki.csv",
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
def load_chembl239_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL239 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Peroxisome proliferator-activated receptor alpha target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1721
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl239_ec50
    >>> dataset = load_chembl239_ec50()
    >>> dataset  # doctest: +SKIP
    (['CCC(Cc1ccc(OC)c(C(=O)NCc2ccc(OCCc3ccccc3)cc2)c1)C(=O)O, ..., 'CC(C)(Oc1ccc(CCOc2ccc(/N=N/c3ccc(Cl)cc3)cc2)cc1)C(=O)O'], \
    array([-3.431, ..., -2.58]))

    >>> dataset = load_chembl239_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  CCC(Cc1ccc(OC)c(C(=O)NCc2ccc(OCCc3ccccc3)cc2)c... -3.431364
    1  CC[C@@H](Cc1ccc(OC)c(C(=O)NCc2ccc(Oc3ccc(F)cc3... -0.964024
    2  CCCCC(Cc1ccc(OC)c(C(=O)NCc2ccc(C(F)(F)F)cc2)c1... -3.000000
    3  CCC(Cc1ccc(OC)c(C(=O)NCCc2ccc(C(F)(F)F)cc2)c1)... -2.869232
    4  CCC(Cc1ccc(OC)c(C(=O)NCc2ccc(OC(F)(F)F)cc2)c1)... -1.633468
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl239_ec50",
        filename="chembl239_ec50.csv",
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
def load_chembl244_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL244 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Coagulation factor x target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  3097
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl244_ki
    >>> dataset = load_chembl244_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(=N)N)cc5c4)c3c2)CC1, ..., 'CC(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CO)C(=O)N[C@H](C=O)CCCNC(=N)N'], \
    array([-0.1139, ..., -2.556]))

    >>> dataset = load_chembl244_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(... -0.113943
    1  CC(=N)N1CCC(Oc2ccc3c(c2)nc(C(C)C)n3Cc2ccc3ccc(... -0.301030
    2  CCC(C)c1nc2cc(OC3CCN(C(C)=N)CC3)ccc2n1Cc1ccc2c... -0.518514
    3  CC1CCN(C(=O)[C@H](Cc2cccc(C(=N)N)c2)NS(=O)(=O)... -3.301000
    4  COC(=O)[C@H]1Cc2ccccc2CN1C(=O)[C@H](Cc1cccc(C(... -4.431000
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl244_ki",
        filename="chembl244_ki.csv",
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
def load_chembl262_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL262 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Glycogen synthase kinase-3 beta target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   856
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl262_ki
    >>> dataset = load_chembl262_ki()
    >>> dataset  # doctest: +SKIP
    (['Cc1nc(N)sc1-c1ccnc(Nc2cccc([N+](=O)[O-])c2)n1, ..., 'CC(C)(C#N)c1cccc(-c2ccnc3[nH]ccc23)n1'], \
    array([-1.301, ..., -2.322]))

    >>> dataset = load_chembl262_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES       Ki
    0      Cc1nc(N)sc1-c1ccnc(Nc2cccc([N+](=O)[O-])c2)n1 -1.30103
    1  Cc1ccc2c(-c3ccnc(Nc4cccc(C(F)(F)F)c4)n3)c(-c3c... -1.30103
    2  Cc1ccc2c(-c3ccnc(Nc4ccc(F)c(F)c4)n3)c(-c3ccc(F... -1.00000
    3  Cc1ccc2c(-c3ccnc(Nc4ccc5c(c4)OCCO5)n3)c(-c3ccc... -1.00000
    4  Cc1ccc2c(-c3ccnc(Nc4ccc(Cl)c(C(F)(F)F)c4)n3)c(... -1.69897
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl262_ki",
        filename="chembl262_ki.csv",
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
def load_chembl264_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL264 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Histamine h3 receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2862
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl264_ki
    >>> dataset = load_chembl264_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(=O)c1ccc(OCCCc2c[nH]cn2)cc1, ..., 'CC(C)(C)c1ccc(OCCCCCCN2CCCCCC2)cc1'], \
    array([-1.94, ..., -2.919]))

    >>> dataset = load_chembl264_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                               SMILES        Ki
    0  CC(=O)c1ccc(OCCCc2c[nH]cn2)cc1 -1.939519
    1       c1ccc(COCCCc2c[nH]cn2)cc1 -0.415974
    2   CC(=O)c1ccc(SCCc2c[nH]cn2)cc1 -0.041393
    3        c1ccc(OCCCc2c[nH]cn2)cc1 -1.431364
    4  CC(=O)c1ccc(SCCCc2c[nH]cn2)cc1 -1.255273
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl264_ki",
        filename="chembl264_ki.csv",
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
def load_chembl287_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL287 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Sigma non-opioid intracellular receptor 1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1328
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl287_ki
    >>> dataset = load_chembl287_ki()
    >>> dataset  # doctest: +SKIP
    (['O=S1(=O)c2ccccc2CCC12CCN(Cc1ccccc1)CC2, ..., 'Cc1[nH]c2cc(C(F)(F)F)ccc2c(=O)c1CN(C)Cc1ccccc1'], \
    array([-1.301, ..., -1.949]))

    >>> dataset = load_chembl287_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0             O=S1(=O)c2ccccc2CCC12CCN(Cc1ccccc1)CC2 -1.301030
    1   COc1ccc(N2C[C@H](CN3CCC(O)(c4ccsc4)CC3)OC2=O)cc1 -1.531479
    2  COc1ccc(N2C[C@H](CN3CCC(O)(c4ccc5c(c4)OCO5)CC3... -1.278754
    3         CNC(=O)CC1Cc2ccccc2C2(CCN(Cc3ccccc3)CC2)O1 -2.230449
    4                OCC1OC2(CCN(Cc3ccccc3)CC2)c2ccccc21 -0.752816
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl287_ki",
        filename="chembl287_ki.csv",
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
def load_chembl1862_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL1862 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Tyrosine-protein kinase abl1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   794
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl1862_ki
    >>> dataset = load_chembl1862_ki()
    >>> dataset  # doctest: +SKIP
    (['Nc1[nH]cnc2nnc(-c3ccc(Cl)cc3)c1-2, ..., 'CCCCNc1ncnc2c1cnn2CC(Cl)c1ccccc1'], \
    array([-2.699, ..., -3.3]))

    >>> dataset = load_chembl1862_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES       Ki
    0                  Nc1[nH]cnc2nnc(-c3ccc(Cl)cc3)c1-2 -2.69897
    1  Cc1ccc(N2NC(=O)/C(=C/c3ccc(-c4ccc(C)c(Cl)c4)o3... -3.69897
    2  O=C1NN(c2ccc(Cl)c(Cl)c2)C(=O)/C1=C\c1cccc(OCc2... -3.00000
    3    O=C1NN(c2ccc(I)cc2)C(=O)/C1=C\c1cc2c(cc1Br)OCO2 -3.39794
    4   O=C1NN(c2ccc(I)cc2)C(=O)/C1=C\c1ccc(N2CCOCC2)cc1 -4.30103
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl1862_ki",
        filename="chembl1862_ki.csv",
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
def load_chembl1871_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL1871 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Androgen receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   659
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl1871_ki
    >>> dataset = load_chembl1871_ki()
    >>> dataset  # doctest: +SKIP
    (['CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccsc1)Oc1ccc(F)cc1-3, ..., 'CN(C[C@](C)(O)C(=O)Nc1ccc(C#N)c(C(F)(F)F)c1)c1ccc(C#N)c(-c2ccccc2)c1'], \
    array([-2.825, ..., -1.892]))

    >>> dataset = load_chembl1871_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccsc1)Oc1ccc(... -2.825426
    1  CCc1ccccc1/C=C1\Oc2ccc(F)cc2-c2ccc3c(c21)C(C)=... -3.201124
    2  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccccc1N(C)C)O... -2.913284
    3  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccccc1)Oc1c(F... -3.163161
    4  CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C[C@H](C)C4... -0.462398
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl1871_ki",
        filename="chembl1871_ki.csv",
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
def load_chembl2034_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL2034 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Glucocorticoid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   750
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl2034_ki
    >>> dataset = load_chembl2034_ki()
    >>> dataset  # doctest: +SKIP
    (['CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccsc1)Oc1ccc(F)cc1-3, ..., 'NS(=O)(=O)C[C@H]1COc2cc(F)ccc2N1C(=O)c1ccc2c(c1)NCCO2'], \
    array([-1.924, ..., -3.1]))

    >>> dataset = load_chembl2034_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccsc1)Oc1ccc(... -1.924279
    1  CCc1ccccc1/C=C1\Oc2ccc(F)cc2-c2ccc3c(c21)C(C)=... -2.431364
    2  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccccc1N(C)C)O... -2.692847
    3  CC1=CC(C)(C)Nc2ccc3c(c21)/C(=C/c1ccccc1)Oc1c(F... -2.506505
    4  CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C[C@H](C)C4... -1.120574
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl2034_ki",
        filename="chembl2034_ki.csv",
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
def load_chembl2047_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL2047 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Bile acid receptor target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   631
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl2047_ec50
    >>> dataset = load_chembl2047_ec50()
    >>> dataset  # doctest: +SKIP
    (['C[C@H](CCC(=O)NCC(=O)O)[C@H]1CC[C@H]2[C@H]3[C@H](CC[C@@]21C)[C@@]1(C)CC[C@@H](O)C[C@H]1C[C@H]3O, ..., 'CC(C)c1onc(-c2c(Cl)cccc2Cl)c1COc1ccc(CNc2ccc(CC(=O)O)cc2)c(Cl)c1'], \
    array([-3.477, ..., -2.973]))

    >>> dataset = load_chembl2047_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  C[C@H](CCC(=O)NCC(=O)O)[C@H]1CC[C@H]2[C@H]3[C@... -3.477121
    1  C[C@H](CCC(=O)O)C1CC[C@H]2[C@H]3[C@H](CC[C@]12... -2.875061
    2  C[C@H](CCC(=O)NCCS(=O)(=O)O)[C@H]1CC[C@H]2[C@H... -3.477121
    3  CCC[C@@H]1C2C[C@H](O)CC[C@]2(C)[C@H]2CC[C@]3(C... -3.045323
    4  CC(C)c1onc(-c2c(Cl)cccc2Br)c1COc1ccc(/C=C/c2cc... -1.079181
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl2047_ec50",
        filename="chembl2047_ec50.csv",
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
def load_chembl2147_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL2147 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Serine/threonine-protein kinase pim-1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1456
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl2147_ki
    >>> dataset = load_chembl2147_ki()
    >>> dataset  # doctest: +SKIP
    (['FC(F)(F)c1cccc(-c2nnc3ccc(NC4CCCCC4)cn23)c1, ..., 'NC(=O)c1cc(Cl)c2c(Cl)c(C#CC3CNCCO3)n([C@H]3CCCNC3)c2n1'], \
    array([-1.041, ..., 0.04576]))

    >>> dataset = load_chembl2147_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0        FC(F)(F)c1cccc(-c2nnc3ccc(NC4CCCCC4)cn23)c1 -1.041393
    1         Cc1ccc2[nH]c(=O)c(CC(=O)O)c(-c3ccccc3)c2c1 -3.653213
    2            O=C(O)c1cccc(Nc2nc(-c3ccc(O)cc3O)cs2)c1 -3.531479
    3       O=C(O)c1cccc2c(-c3ccccc3)c(-c3ccccc3)[nH]c12 -2.740363
    4  CCc1ccc(C2C(C(C)=O)=C(O)C(=O)N2CCc2c[nH]c3cccc... -3.322219
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl2147_ki",
        filename="chembl2147_ki.csv",
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
def load_chembl2835_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL2835 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Tyrosine-protein kinase jak1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   615
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl2835_ki
    >>> dataset = load_chembl2835_ki()
    >>> dataset  # doctest: +SKIP
    (['C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12, ..., 'Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Nc1cccc(S(=O)(=O)NC(C)(C)C)c1'], \
    array([0.1549, ..., -2.021]))

    >>> dataset = load_chembl2835_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]c...  0.154902
    1  C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1n1cnc2cnc3[nH]cc...  0.301030
    2  C[C@@H]1CCN(Cc2ccccc2)C[C@@H]1N(C)c1ncnc2[nH]c... -2.785330
    3  C[C@@H]1CCN(Cc2ccccc2)C[C@@H]1n1cnc2cnc3[nH]cc... -1.079181
    4      N#CCC(=O)N1CCC[C@@H](n2cnc3cnc4[nH]ccc4c32)C1  0.397940
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl2835_ki",
        filename="chembl2835_ki.csv",
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
def load_chembl2971_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL2971 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Tyrosine-protein kinase jak2 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   976
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl2971_ki
    >>> dataset = load_chembl2971_ki()
    >>> dataset  # doctest: +SKIP
    (['NC(=O)Nc1sc(-c2ccc(F)cc2)cc1C(N)=O, ..., 'Cc1cc(Nc2nc(N[C@@H](C)c3ccc(F)cc3)c(C#N)cc2F)n[nH]1'], \
    array([-0.699, ..., 0.3468]))

    >>> dataset = load_chembl2971_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0                 NC(=O)Nc1sc(-c2ccc(F)cc2)cc1C(N)=O -0.698970
    1  O[C@H]1CC[C@H](Nc2ccc3nnc(-c4cccc(C(F)(F)F)c4)... -3.380211
    2                        c1ccc(-c2ncnc3[nH]ccc23)cc1 -2.683947
    3                      Clc1cnc2[nH]cc(-c3ccccc3)c2c1 -2.414973
    4                    CCC1Nc2ccccc2-c2ccnc3[nH]cc1c23 -3.230449
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl2971_ki",
        filename="chembl2971_ki.csv",
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
def load_chembl3979_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL3979 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Peroxisome proliferator-activated receptor delta target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1125
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl3979_ec50
    >>> dataset = load_chembl3979_ec50()
    >>> dataset  # doctest: +SKIP
    (['CCC(Cc1ccc(OC)c(C(=O)NCCc2ccc(C(F)(F)F)cc2)c1)C(=O)O, ..., 'CC(C)c1onc(-c2c(Cl)cccc2Cl)c1COc1ccc(CNc2ccc(CC(=O)O)cc2)c(Cl)c1'], \
    array([-3.176, ..., -3.176]))

    >>> dataset = load_chembl3979_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  CCC(Cc1ccc(OC)c(C(=O)NCCc2ccc(C(F)(F)F)cc2)c1)... -3.176091
    1  CCC(Cc1ccc(OC)c(C(=O)NCc2ccc(OC(F)(F)F)cc2)c1)... -2.954243
    2    CCC(Cc1ccc(OC)c(CCCc2ccc(C(F)(F)F)cc2)c1)C(=O)O -2.806180
    3  CCSC(Cc1ccc(OC)c(C(=O)NCc2ccc(C(F)(F)F)cc2)c1)... -3.477121
    4  CCOC(Cc1ccc(OC)c(C(=O)NCc2ccc(C(F)(F)F)cc2)c1)... -3.477121
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl3979_ec50",
        filename="chembl3979_ec50.csv",
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
def load_chembl4005_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL4005 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit alpha isoform target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   960
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl4005_ki
    >>> dataset = load_chembl4005_ki()
    >>> dataset  # doctest: +SKIP
    (['COC[C@H]1OC(=O)c2coc3c2[C@@]1(C)C1=C(C3=O)[C@@H]2CCC(=O)[C@@]2(C)C[C@H]1OC(C)=O, ..., 'CC(C)n1nc(-c2ccc3oc(N)nc3c2)c2c(N)ncnc21'], \
    array([-2.079, ..., -1.447]))

    >>> dataset = load_chembl4005_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  COC[C@H]1OC(=O)c2coc3c2[C@@]1(C)C1=C(C3=O)[C@@... -2.079181
    1              O=c1cc(N2CCOCC2)oc2c(-c3ccccc3)cccc12 -3.778151
    2  CS(=O)(=O)N1CCN(Cc2cc3nc(-c4cccc5[nH]ncc45)nc(... -0.806180
    3          COc1ccc(NC(=O)c2c(C)ccc3c(N)nc(C)nc23)cn1 -2.000000
    4          COc1ccc(NC(=O)c2cc(C)cc3c(N)nc(C)nc23)cn1  0.301030
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl4005_ki",
        filename="chembl4005_ki.csv",
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
def load_chembl4203_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL4203 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Dual specificity protein kinase clk4 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   731
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl4203_ki
    >>> dataset = load_chembl4203_ki()
    >>> dataset  # doctest: +SKIP
    (['O=c1[nH]cnc2c1sc1c(Cl)ccc(Cl)c12, ..., 'O=C(c1cccc(-c2cnc3[nH]ccc3c2)c1)N1CCOCC1'], \
    array([-1.977, ..., -3.8]))

    >>> dataset = load_chembl4203_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0                   O=c1[nH]cnc2c1sc1c(Cl)ccc(Cl)c12 -1.976808
    1  Nc1ncnc2onc(-c3ccc(NC(=O)Nc4cccc(C(F)(F)F)c4)c... -2.400002
    2            O=c1[nH]cnc2c(-c3ccccc3)c(C(F)(F)F)sc12 -3.299999
    3                O=C1Nc2ccccc2Nc2cc(-c3ccncc3F)ccc21 -1.400020
    4  Cc1cc(N2CCOCC2)cc2[nH]c(-c3c(NCC(O)c4cccc(Cl)c... -1.700011
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl4203_ki",
        filename="chembl4203_ki.csv",
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
def load_chembl4616_ec50(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL4616 EC50 dataset.

    The task is to predict the half maximal effective concentration (EC50) of molecules against the Growth hormone secretagogue receptor type 1 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   682
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl4616_ec50
    >>> dataset = load_chembl4616_ec50()
    >>> dataset  # doctest: +SKIP
    (['CCCCCCCC(=O)OC[C@H](NC(=O)CN)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O, ..., 'CC(=O)N1CCC[C@H](NC(=O)[C@H]2CN(S(=O)(=O)c3ccccc3)C[C@@H]2NC(=O)c2cc(-c3ccccc3Cl)on2)C1'], \
    array([-1.857, ..., -2.111]))

    >>> dataset = load_chembl4616_ec50(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES      EC50
    0  CCCCCCCC(=O)OC[C@H](NC(=O)CN)C(=O)N[C@@H](CO)C... -1.857332
    1  CCCCCCCC(=O)OC[C@H](NC(=O)CNC(=O)[C@@H](N)CCCN... -0.147985
    2  CC(C)(N)C(=O)N[C@H](COCc1ccccc1)C(=O)N1CCC2(CC...  0.072578
    3  NC(=O)CN(CCc1ccccc1)C(=O)[C@@H](Cc1ccc2ccccc2c...  0.468521
    4  CC(C)N(CCNC(=O)C1c2ccc(Oc3cccc(F)c3)cc2CCN1C(=... -0.633468
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl4616_ec50",
        filename="chembl4616_ec50.csv",
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
def load_chembl4792_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ChEMBL4792 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Orexin receptor type 2 target [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1471
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods"
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculeace import load_chembl4792_ki
    >>> dataset = load_chembl4792_ki()
    >>> dataset  # doctest: +SKIP
    (['CC1(C)OC[C@H](NC(=O)Nc2ccc(Br)cc2Cl)[C@H](c2ccccc2)O1, ..., 'CC(/C=C/c1ccccc1)=N/Nc1nc(Nc2ccccc2)nc(-n2nc(C)cc2C)n1'], \
    array([-0.8, ..., -4.25]))

    >>> dataset = load_chembl4792_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES        Ki
    0  CC1(C)OC[C@H](NC(=O)Nc2ccc(Br)cc2Cl)[C@H](c2cc... -0.800029
    1  Cc1cc(Br)ccc1NC(=O)N[C@H]1COC(C)(C)O[C@H]1c1cc... -1.599992
    2  Cc1ccc(Cl)c(NC(=O)N[C@H]2COC(C)(C)O[C@H]2c2ccc... -1.800029
    3  Cc1ccc(NC(=O)N[C@H]2COC(C)(C)O[C@H]2c2ccccc2)c... -2.099991
    4  CC1(C)OC[C@H](NC(=O)Nc2cc(Cl)ccc2Cl)[C@H](c2cc... -1.800029
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl4792_ki",
        filename="chembl4792_ki.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
