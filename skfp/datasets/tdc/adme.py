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
def load_b3db_classification(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the classification subset of Blood-Brain-Barrier dataset.

    The task is to classify molecules as either BBB permeable (BBB+) or BBB non-permeable (BBB-) [1]_ [2]_.
    The BBB permeability is measured by the logarithm of the brain-plasma concentration ratio:

    .. math::

        \log BB = \log \frac{C_{brain}}{C_{blood}}

    Where :math:`C_{brain}` and :math:`C_{blood}` are concentrations in brain and blood, respectively.
    The molecules with :math:`\log BB` greater than 0 make up the positive class.

    This dataset should not be confused with BBBP dataset from MoleculeNet.

    See also :py:func:`load_b3db_regression`

    This dataset is a part of "distribution" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     6167
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
    .. [1] `Meng, F., Xi, Y., Huang, J. & Ayers, P. W.
        "A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors"
        Sci Data 8, 289 (2021).
        <https://doi.org/10.1038/s41597-021-01069-5>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_b3db_classification",
        filename="tdc_b3db_classification.csv",
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
def load_b3db_regression(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the regression subset of Blood-Brain-Barrier dataset.

    The task is to predict the BBB permeability [1]_ [2]_.
    The BBB permeability is measured by the logarithm of the brain-plasma concentration ratio:

    .. math::

        \log BB = \log \frac{C_{brain}}{C_{blood}}

    Where :math:`C_{brain}` and :math:`C_{blood}` are concentrations in brain and blood, respectively.
    This dataset should not be confused with BBBP dataset from MoleculeNet.

    See also :py:func:`load_b3db_classification`

    This dataset is a part of "distribution" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                      942
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
    .. [1] `Meng, F., Xi, Y., Huang, J. & Ayers, P. W.
        "A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors"
        Sci Data 8, 289 (2021).
        <https://doi.org/10.1038/s41597-021-01069-5>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_b3db_regression",
        filename="tdc_b3db_regression.csv",
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
def load_bioavailability_ma(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Bioavailability dataset.

    The task is to predict the oral bioavailability.
    It is defined as "the rate and extent to which the active ingredient or active moiety
    is absorbed from a drug product and becomes available at the site of action" [1]_ [2]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      640
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
    .. [1] `Ma, Chang-Ying, et al.
        "Prediction models of human plasma protein binding rate
        and oral bioavailability derived by using GA–CG–SVM method"
        Journal of Pharmaceutical and Biomedical Analysis 47.4-5 (2008): 677-682.
        <https://doi.org/10.1016/j.jpba.2008.03.023>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_bioavailability_ma",
        filename="tdc_bioavailability_ma.csv",
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
def load_caco2_wang(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Caco-2 dataset.

    The task is to predict the rate at which drug passes through Caco-2 cells
    that serve as in vitro simulation of human intestinal tissue [1]_ [2]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                      910
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
    .. [1] `Wang, NN, et al.
        "ADME Properties Evaluation in Drug Discovery: Prediction of Caco-2 Cell Permeability Using a Combination of NSGA-II and Boosting"
        Journal of Chemical Information and Modeling 2016 56 (4), 763-773
        <https://doi.org/10.1021/acs.jcim.5b00642>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_caco2_wang",
        filename="tdc_caco2_wang.csv",
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
def load_clearance_microsome_az(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the microsome subset of Clearance AstraZeneca dataset.

    The task is to predict drug clearance.
    It is defined as the volume of plasma cleared of a drug over a specified time period
    and it measures the rate at which the active drug is removed from the body [1]_ [2]_ [3]_.
    Many studies [2]_ show various clearance outcomes of experiments performed with
    human hepatocytes (HHEP) and human liver microsomes (HLM) which are two main
    in vitro systems used in metabolic stability and inhibition studies.
    This subset od the Clearance dataset includes measurements from microsome studies.

    This dataset is a part of "excretion" subset of ADME tasks.

    ==================  ====================
    Tasks                                  1
    Task type                     regression
    Total samples                       1102
    Recommended split               scaffold
    Recommended metric  Spearman correlation
    ==================  ====================

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
    .. [1] `AstraZeneca.
        "Experimental in vitro Dmpk and physicochemical data on a set of publicly disclosed compounds"
        (2016)
        <https://www.ebi.ac.uk/chembl/explore/document/CHEMBL3301361>`_

    .. [2] `Di, Li, et al.
        "Mechanistic insights from comparing intrinsic clearance values
        between human liver microsomes and hepatocytes to guide drug design"
        European Journal of Medicinal Chemistry 57 (2012): 441-448.
        <https://doi.org/10.1016/j.ejmech.2012.06.043>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_clearance_microsome_az",
        filename="tdc_clearance_microsome_az.csv",
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
def load_cyp1a2_veith(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP1A2 subset of CYP P450 Veith dataset.

    The CYP P450 genes are involved in the formation and breakdown (metabolism)
    of various molecules and chemicals within cells.
    The task is to predict CYP1A2 inhibition.
    CYP1A2 localizes to the endoplasmic reticulum and its expression is induced by some polycyclic
    aromatic hydrocarbons (PAHs), some of which are found in cigarette smoke.
    It is able to metabolize some PAHs to carcinogenic intermediates.
    Other xenobiotic substrates for this enzyme include caffeine, aflatoxin B1, and acetaminophen [1]_ [2]_.

    This dataset is a part of "metabolism" subset of ADME tasks.

    All CYP P450 Veith subsets:

        - :py:func:`load_cyp1a2_veith`
        - :py:func:`load_cyp2c9_veith`
        - :py:func:`load_cyp2c19_veith`
        - :py:func:`load_cyp2d6_veith`
        - :py:func:`load_cyp3a4_veith`

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    12579
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Veith, Henrike et al.
        "Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries"
        Nature Biotechnology vol. 27,11 (2009): 1050-5.
        <https://doi.org/10.1038/nbt.1581>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp1a2_veith",
        filename="tdc_cyp1a2_veith.csv",
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
def load_cyp2c9_substrate_carbonmangels(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP2C9 subset of Substrate Carbon-Mangels dataset.

    CYP2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds [1]_ [2]_ [3]_.
    Substrates are drugs that are metabolized by the enzyme.
    The task is to predict whether a molecule is a substrate to CYP2C9.

    All Substrate Carbon-Mangels subsets:

        - :py:func:`load_cyp2c9_substrate_carbonmangels`
        - :py:func:`load_cyp2d6_substrate_carbonmangels`
        - :py:func:`load_cyp3a4_substrate_carbonmangels`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      669
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Carbon‐Mangels, Miriam, and Michael C. Hutter.
        "Selecting relevant descriptors for classification by bayesian estimates:
        a comparison with decision trees and support vector machines approaches for disparate data sets"
        Molecular Informatics 30.10 (2011): 885-895.
        <https://doi.org/10.1002/minf.201100069>`_

    .. [2] `Cheng, Feixiong, et al.
        "admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties"
        J. Chem. Inf. Model. 2012, 52, 11, 3099–3105
        <https://doi.org/10.1002/minf.201100069>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp2c9_substrate_carbonmangels",
        filename="tdc_cyp2c9_substrate_carbonmangels.csv",
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
def load_cyp2c9_veith(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP2C9 subset of CYP P450 Veith dataset.

    The CYP P450 genes are involved in the formation and breakdown (metabolism)
    of various molecules and chemicals within cells.
    The task is to predict CYP2C9 inhibition.
    CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds [1]_ [2]_.

    All CYP P450 Veith subsets:

        - :py:func:`load_cyp1a2_veith`
        - :py:func:`load_cyp2c9_veith`
        - :py:func:`load_cyp2c19_veith`
        - :py:func:`load_cyp2d6_veith`
        - :py:func:`load_cyp3a4_veith`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    12092
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Veith, Henrike et al.
        "Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries"
        Nature Biotechnology vol. 27,11 (2009): 1050-5.
        <https://doi.org/10.1038/nbt.1581>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp2c9_veith",
        filename="tdc_cyp2c9_veith.csv",
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
def load_cyp2c19_veith(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP2C19 subset of CYP P450 Veith dataset.

    The CYP P450 genes are involved in the formation and breakdown (metabolism)
    of various molecules and chemicals within cells.
    The task is to predict CYP2C19 inhibition.
    CYP2C19 gene provides instructions for making an enzyme called the endoplasmic reticulum,
    which is involved in protein processing and transport [1]_ [2]_.

    All CYP P450 Veith subsets:

        - :py:func:`load_cyp1a2_veith`
        - :py:func:`load_cyp2c9_veith`
        - :py:func:`load_cyp2c19_veith`
        - :py:func:`load_cyp2d6_veith`
        - :py:func:`load_cyp3a4_veith`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    12665
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Veith, Henrike et al.
        "Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries"
        Nature Biotechnology vol. 27,11 (2009): 1050-5.
        <https://doi.org/10.1038/nbt.1581>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp2c19_veith",
        filename="tdc_cyp2c19_veith.csv",
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
def load_cyp2d6_substrate_carbonmangels(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP2D6 subset of Substrate Carbon-Mangels dataset.

    CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system,
    including the substantia nigra [1]_ [2]_ [3]_.
    Substrates are drugs that are metabolized by the enzyme.
    The task is to predict whether a molecule is a substrate to CYP2D6.

    All Substrate Carbon-Mangels subsets:

        - :py:func:`load_cyp2c9_substrate_carbonmangels`
        - :py:func:`load_cyp2d6_substrate_carbonmangels`
        - :py:func:`load_cyp3a4_substrate_carbonmangels`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      667
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Carbon‐Mangels, Miriam, and Michael C. Hutter.
        "Selecting relevant descriptors for classification by bayesian estimates:
        a comparison with decision trees and support vector machines approaches for disparate data sets"
        Molecular Informatics 30.10 (2011): 885-895.
        <https://doi.org/10.1002/minf.201100069>`_

    .. [2] `Cheng, Feixiong, et al.
        "admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties"
        J. Chem. Inf. Model. 2012, 52, 11, 3099–3105
        <https://doi.org/10.1002/minf.201100069>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp2d6_substrate_carbonmangels",
        filename="tdc_cyp2d6_substrate_carbonmangels.csv",
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
def load_cyp2d6_veith(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP2D6 subset of CYP P450 Veith dataset.

    The CYP P450 genes are involved in the formation and breakdown (metabolism)
    of various molecules and chemicals within cells.
    The task is to predict CYP2D6 inhibition.
    CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system,
    including the substantia nigra [1]_ [2]_.

    All CYP P450 Veith subsets:

        - :py:func:`load_cyp1a2_veith`
        - :py:func:`load_cyp2c9_veith`
        - :py:func:`load_cyp2c19_veith`
        - :py:func:`load_cyp2d6_veith`
        - :py:func:`load_cyp3a4_veith`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    13130
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Veith, Henrike et al.
        "Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries"
        Nature Biotechnology vol. 27,11 (2009): 1050-5.
        <https://doi.org/10.1038/nbt.1581>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp2d6_veith",
        filename="tdc_cyp2d6_veith.csv",
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
def load_cyp3a4_substrate_carbonmangels(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP3A4 subset of Substrate Carbon-Mangels dataset.

    CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine.
    It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs,
    so that they can be removed from the body [1]_ [2]_ [3]_.
    Substrates are drugs that are metabolized by the enzyme.
    The task is to predict whether a molecule is a substrate to CYP3A4.

    All Substrate Carbon-Mangels subsets:

        - :py:func:`load_cyp2c9_substrate_carbonmangels`
        - :py:func:`load_cyp2d6_substrate_carbonmangels`
        - :py:func:`load_cyp3a4_substrate_carbonmangels`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      670
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
    .. [1] `Carbon‐Mangels, Miriam, and Michael C. Hutter.
        "Selecting relevant descriptors for classification by bayesian estimates:
        a comparison with decision trees and support vector machines approaches for disparate data sets"
        Molecular Informatics 30.10 (2011): 885-895.
        <https://doi.org/10.1002/minf.201100069>`_

    .. [2] `Cheng, Feixiong, et al.
        "admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties"
        J. Chem. Inf. Model. 2012, 52, 11, 3099–3105
        <https://doi.org/10.1002/minf.201100069>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp3a4_substrate_carbonmangels",
        filename="tdc_cyp3a4_substrate_carbonmangels.csv",
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
def load_cyp3a4_veith(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the CYP3A4 subset of CYP P450 Veith dataset.

    The CYP P450 genes are involved in the formation and breakdown (metabolism)
    of various molecules and chemicals within cells.
    The task is to predict CYP3A4 inhibition.
    CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine.
    It oxidizes small foreign organic molecules (xenobiotics),
    such as toxins or drugs, so that they can be removed from the body [1]_ [2]_.

    All CYP P450 Veith subsets:

        - :py:func:`load_cyp1a2_veith`
        - :py:func:`load_cyp2c9_veith`
        - :py:func:`load_cyp2c19_veith`
        - :py:func:`load_cyp2d6_veith`
        - :py:func:`load_cyp3a4_veith`

    This dataset is a part of "metabolism" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                    12328
    Recommended split             scaffold
    Recommended metric               AUPRC
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
    .. [1] `Veith, Henrike et al.
        "Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries"
        Nature Biotechnology vol. 27,11 (2009): 1050-5.
        <https://doi.org/10.1038/nbt.1581>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_cyp3a4_veith",
        filename="tdc_cyp3a4_veith.csv",
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
def load_half_life_obach(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Half Life Obach dataset.

    The task is to predict the half life of a drug.
    It is defined as a duration for the concentration of the drug in the body to be reduced by half.
    It measures the duration of actions of a drug [1]_ [2]_.

    This dataset is a part of "excretion" subset of ADME tasks.

    ==================  ====================
    Tasks                                  1
    Task type                     regression
    Total samples                        667
    Recommended split               scaffold
    Recommended metric  Spearman correlation
    ==================  ====================

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
    .. [1] `Obach, R. Scott, Franco Lombardo, and Nigel J. Waters.
        “Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 670 Drug Compounds”
        Drug Metabolism and Disposition 36.7 (2008): 1385-1405.
        <https://doi.org/10.1124/dmd.108.020479>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_half_life_obach",
        filename="tdc_half_life_obach.csv",
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
def load_hia_hou(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Human Intestinal Absorption dataset.

    The task is to predict whether a drug is well absorbed
    via the human intestine. It is relevant for oral drug design [1]_ [2]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      578
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
    .. [1] `Hou T et al.
        "ADME evaluation in drug discovery. 7. Prediction of oral absorption by correlation and classification"
        J Chem Inf Model. 2007;47(1):208-218.
        <https://doi.org/10.1021/ci600343x>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_hia_hou",
        filename="tdc_hia_hou.csv",
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
def load_hlm(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the human subset of Human/Rat Liver Microsomal Stability dataset.

    Liver microsomal stability or hepatic metabolic stability
    is an important property considered for the screening of drug candidates.
    The task is to determine whether the drug is stable or not [1]_ [2]_.
    This subset relates to human liver microsomes.

    See also :py:func:`load_rlm`

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     6013
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
    .. [1] `Longqiang, Li., Zhou, Lu., Guixia, Liu., Yun, Tang., Weihua, Li.
        "In Silico Prediction of Human and Rat Liver Microsomal Stability via Machine Learning Methods"
        Chem. Res. Toxicol. 2022, 35, 9, 1614–1624
        <https://doi.org/10.1021/acs.chemrestox.2c00207>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_hlm",
        filename="tdc_hlm.csv",
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
def load_pampa_approved_drugs(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the approved drugs subset of PAMPA dataset.

    PAMPA (parallel artificial membrane permeability assay) is an assay
    to evaluate drug permeability across the cellular membrane.
    The task models only the passive membrane diffusion [1]_ [2]_.
    This is the "approved drugs" subset that includes 142 marketed-approved drugs assessed by NCATS [1]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      142
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
    .. [1] `Siramshetty, V.B., Shah, P., et al.
        "Validating ADME QSAR Models Using Marketed Drugs"
        SLAS Discovery 2021 Dec;26(10):1326-1336.
        <https://doi.org/10.1177/24725552211017520>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_pampa_approved_drugs",
        filename="tdc_pampa_approved_drugs.csv",
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
def load_pampa_ncats(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the NCATS subset of PAMPA dataset.

    PAMPA (parallel artificial membrane permeability assay) is an assay
    to evaluate drug permeability across the cellular membrane.
    The task models only the passive membrane diffusion [1]_ [2]_.
    This is "NCATS" subset of the dataset created at National Center for Advancing Translational Sciences (NCATS).

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     2034
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
    .. [1] `Siramshetty, V.B., Shah, P., et al.
        "Validating ADME QSAR Models Using Marketed Drugs"
        SLAS Discovery 2021 Dec;26(10):1326-1336.
        <https://doi.org/10.1177/24725552211017520>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_pampa_ncats",
        filename="tdc_pampa_ncats.csv",
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
def load_pgp_broccatelli(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the P-gp (P-glycoprotein) Inhibition dataset.

    The task is to predict whether a molecule inhibits the P-glycoprotein.
    P-glycoprotein (P-gp) is an ABC transporter protein involved in intestinal absorption, drug metabolism,
    and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety [1]_ [2]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     1218
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
    .. [1] `Broccatelli et al.
        "A Novel Approach for Predicting P-Glycoprotein (ABCB1) Inhibition Using Molecular Interaction Fields"
        Journal of Medicinal Chemistry, 2011 54 (6), 1740-1751
        <https://doi.org/10.1021/jm101421d>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_pgp_broccatelli",
        filename="tdc_pgp_broccatelli.csv",
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
def load_ppbr_az(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the PPBR (Plasma Protein Binding Rate) AstraZeneca dataset.

    The task is to predict human plasma protein binding rate (PPBR) [1]_ [2]_.
    PPBR is expressed as the percentage of a drug bound to plasma proteins in the blood.
    This rate strongly affects the drug delivery efficiency.
    The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions.

    This dataset is a part of "distribution" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                     1614
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
    .. [1] `AstraZeneca.
        "Experimental in vitro Dmpk and physicochemical data on a set of publicly disclosed compounds"
        (2016)
        <https://www.ebi.ac.uk/chembl/explore/document/CHEMBL3301361>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_ppbr_az",
        filename="tdc_ppbr_az.csv",
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
def load_rlm(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the rat subset of Human/Rat Liver Microsomal Stability dataset.

    Liver microsomal stability or hepatic metabolic stability
    is an important property considered for the screening of drug candidates.
    The task is to determine whether the drug is stable or not [1]_ [2]_.
    This subset relates to rat liver microsomes.

    See also :py:func:`load_rlm`

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     6013
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
    .. [1] `Longqiang, Li., Zhou, Lu., Guixia, Liu., Yun, Tang., Weihua, Li.
        "In Silico Prediction of Human and Rat Liver Microsomal Stability via Machine Learning Methods"
        Chem. Res. Toxicol. 2022, 35, 9, 1614–1624
        <https://doi.org/10.1021/acs.chemrestox.2c00207>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_rlm",
        filename="tdc_rlm.csv",
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
def load_solubility_aqsoldb(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Solubility AqSolDB dataset.

    The task is to predict the aqeuous solubility - a measure drug's ability to dissolve in water.
    Poor water solubility could lead to slow drug absorptions,
    inadequate bioavailablity and even induce toxicity [1]_ [2]_.

    This dataset is a part of "absorption" subset of ADME tasks.

    ==================   =================
    Tasks                                1
    Task type                   regression
    Total samples                     9982
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
    .. [1] `Sorkun, M.C., Khetan, A. & Er, S.
        "AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds"
        Sci Data 6, 143 (2019).
        <https://doi.org/10.1038/s41597-019-0151-1>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_solubility_aqsoldb",
        filename="tdc_solubility_aqsoldb.csv",
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
def load_vdss_lombardo(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Volume of Distribution at Steady State dataset.

    The task is to predict the volume of distribution at steady state (VDss) that measures
    the degree of drug's concentration in body tissue compared to concentration in blood [1]_ [2]_.

    This dataset is a part of "distribution" subset of ADME tasks.

    ==================  ====================
    Tasks                                  1
    Task type                     regression
    Total samples                       1130
    Recommended split               scaffold
    Recommended metric  Spearman correlation
    ==================  ====================

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
    .. [1] `Lombardo, Franco, and Yankang Jing.
        "In Silico Prediction of Volume of Distribution in Humans.
        Extensive Data Set and the Exploration of Linear and Nonlinear Methods
        Coupled with Molecular Interaction Fields Descriptors"
        Journal of Chemical Information and Modeling 56.10 (2016): 2042-2052.
        <https://doi.org/10.1021/acs.jcim.6b00044>`_

    .. [2] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_vdss_lombardo",
        filename="tdc_vdss_lombardo.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
