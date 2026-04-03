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
def load_bace(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the BACE dataset.

    The task is to predict binding results for a set of inhibitors of human
    β-secretase 1 (BACE-1) [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                  1513
    Recommended split          scaffold
    Recommended metric            AUROC
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
    .. [1] `Govindan Subramanian et al.
        "Computational Modeling of β-Secretase 1 (BACE-1) Inhibitors Using
        Ligand Based Approaches"
        J. Chem. Inf. Model. 2016, 56, 10, 1936-1949
        <https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_bace
    >>> dataset = load_bace()
    >>> dataset  # doctest: +SKIP
    (['O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C', ..., 'Clc1cc2nc(n(c2cc1)CCCC(=O)NCC1CC1)N'], \
array([1, 1, 1, ..., 0, 0, 0]))

    >>> dataset = load_bace(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES  label
    0  O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2c...      1
    1  Fc1cc(cc(F)c1)C[C@H](NC(=O)[C@@H](N1CC[C@](NC(...      1
    2  S1(=O)(=O)N(c2cc(cc3c2n(cc3CC)CC1)C(=O)N[C@H](...      1
    3  S1(=O)(=O)C[C@@H](Cc2cc(O[C@H](COCC)C(F)(F)F)c...      1
    4  S1(=O)(=O)N(c2cc(cc3c2n(cc3CC)CC1)C(=O)N[C@H](...      1

    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_BACE", filename="bace.csv", verbose=verbose
    )
    return df if as_frame else get_mol_strings_and_labels(df)
