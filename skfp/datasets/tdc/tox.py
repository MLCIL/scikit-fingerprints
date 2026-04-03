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
def load_ames(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the AMES dataset.

    The task is to predict mutagenicity of drugs, i.e. potential to induce
    genetic alterations [1]_ [2]_. This data comes from a standardized Ames test,
    which is a short-term bacterial reverse mutation assay.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     7278
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
    .. [1] `Xu, Congying, et al.
        "In silico Prediction of Chemical Ames Mutagenicity"
        Journal of Chemical Information and Modeling 52.11 (2012): 2840-2847
        <https://doi.org/10.1021/ci300400a>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_ames",
        filename="tdc_ames.csv",
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
def load_carcinogens_lagunin(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Carcinogens dataset.

    The task is to predict whether the drug is a carcinogen [1]_ [2]_ [3]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      280
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
    .. [1] `Lagunin, Alexey, et al.
        "Computer-Aided Prediction of Rodent Carcinogenicity by PASS and CISOC-PSCT"
        QSAR & Combinatorial Science 28.8 (2009): 806-810
        <https://doi.org/10.1002/qsar.200860192>`_

    .. [2] `Cheng, Feixiong, et al.
        "admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties"
        Journal of Chemical Information and Modeling 52.11 (2012): 2840-2847
        <https://doi.org/10.1021/ci300367a>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_carcinogens_lagunin",
        filename="tdc_carcinogens_lagunin.csv",
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
def load_dili(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the DILI (Drug Induced Liver Injury) dataset.

    DILI (Drug-Induced Liver Injury) is the most frequent cause of safety-related
    drug withdrawal. The task of this dataset is to predict whether a drug can
    cause such liver injury [1]_ [2]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      475
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
    .. [1] `Xu, Youjun, et al.
        "Deep Learning for Drug-Induced Liver Injury"
        Journal of Chemical Information and Modeling 55.10 (2015): 2085-2093
        <https://doi.org/10.1021/acs.jcim.5b00238>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_dili",
        filename="tdc_dili.csv",
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
def load_herg(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the hERG blockers dataset.

    The task is to predict whether the drug blocks ether-à-go-go related gene (hERG),
    crucial for coordination of the heart's beating [1]_ [2]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      655
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
    .. [1] `Wang, Shuangquan, et al.
        "ADMET Evaluation in Drug Discovery. 16. Predicting hERG Blockers by Combining
        Multiple Pharmacophores and Machine Learning Approaches"
        Molecular Pharmaceutics 13.8 (2016): 2855-2866.
        <https://doi.org/10.1021/acs.molpharmaceut.6b00471>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_herg",
        filename="tdc_herg.csv",
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
def load_herg_central_at_1um(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the 1 µM subset of hERG Central dataset.

    The task is to predict the inhibition of ether-à-go-go related gene (hERG),
    crucial for coordination of the heart's beating [1]_ [2]_.

    In this subset, the task is predicting percent inhibition at 1 µM concentration.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                   306893
    Recommended split             scaffold
    Recommended metric                 MAE
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
    .. [1] `Du F, et al.
        "hERGCentral: a large database to store, retrieve, and analyze compound-human Ether-à-go-go
        related gene channel interactions to facilitate cardiotoxicity assessment in drug development"
        Assay Drug Dev Technol. 2011 Dec;9(6):580-8.
        <https://doi.org/10.1089/adt.2011.0425>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_herg_central_at_1um",
        filename="tdc_herg_central_at_1um.csv",
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
def load_herg_central_at_10um(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the 10 µM subset of hERG Central dataset.

    The task is to predict the inhibition of ether-à-go-go related gene (hERG),
    crucial for coordination of the heart's beating [1]_ [2]_.

    In this subset, the task is predicting percent inhibition at 10 µM concentration.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                   306893
    Recommended split             scaffold
    Recommended metric                 MAE
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
    .. [1] `Du F, et al.
        "hERGCentral: a large database to store, retrieve, and analyze compound-human Ether-à-go-go
        related gene channel interactions to facilitate cardiotoxicity assessment in drug development"
        Assay Drug Dev Technol. 2011 Dec;9(6):580-8.
        <https://doi.org/10.1089/adt.2011.0425>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_herg_central_at_10um",
        filename="tdc_herg_central_at_10um.csv",
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
def load_herg_central_inhib(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the inhibition subset of hERG Central dataset.

    The task is to predict the inhibition of ether-à-go-go related gene (hERG),
    crucial for coordination of the heart's beating [1]_ [2]_.

    In this subset, the task is predicting the binary inhibition ability, i.e.
    if percentage inhibition at 10 µM is smaller than -50.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                   306893
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
    .. [1] `Du F, et al.
        "hERGCentral: a large database to store, retrieve, and analyze compound-human Ether-à-go-go
        related gene channel interactions to facilitate cardiotoxicity assessment in drug development"
        Assay Drug Dev Technol. 2011 Dec;9(6):580-8.
        <https://doi.org/10.1089/adt.2011.0425>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_herg_central_inhib",
        filename="tdc_herg_central_inhib.csv",
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
def load_herg_karim(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the hERG Karim dataset.

    The task is to predict whether the drug blocks ether-à-go-go related gene (hERG),
    crucial for coordination of the heart's beating [1]_ [2]_.

    This dataset is a binary classification task, with molecules defined as hERG (<10 µM)
    and non-hERG (>=10 µM) blockers.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    13445
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
    .. [1] `Karim, A., et al.
        "CardioTox net: a robust predictor for hERG channel blockade based on
        deep learning meta-feature ensembles"
        Journal of Cheminformatics 13, 60 (2021).
        <https://doi.org/10.1186/s13321-021-00541-z>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_herg_karim",
        filename="tdc_herg_karim.csv",
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
def load_ld50_zhu(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Acute Toxicity LD50 dataset.

    Acute toxicity LD50 measures the most conservative dose that
    can lead to lethal adverse effects [1]_ [2]_.
    The regression task is to predict the acute toxicity of drugs.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                     7385
    Recommended split             scaffold
    Recommended metric                 MAE
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
    .. [1] `Zhu, Hao, et al.
        "Quantitative Structure−Activity Relationship Modeling of Rat Acute Toxicity by Oral Exposure"
        Chemical Research in Toxicology 22.12 (2009): 1913-1921.
        <https://doi.org/10.1021/tx900189p>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_ld50_zhu",
        filename="tdc_ld50_zhu.csv",
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
def load_skin_reaction(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Skin Reaction dataset.

    The task is to predict whether the drug can cause immune reaction
    that leads to skin sensitization [1]_ [2]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      404
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
    .. [1] `Alves, Vinicius M., et al.
        "Predicting chemically-induced skin reactions. Part I: QSAR models of
        skin sensitization and their application to identify potentially hazardous compounds"
        Toxicology and Applied Pharmacology 284.2 (2015): 262-272.
        <https://doi.org/10.1016/j.taap.2014.12.014>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_skin_reaction",
        filename="tdc_skin_reaction.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
