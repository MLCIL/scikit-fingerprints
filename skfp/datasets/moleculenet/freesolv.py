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
def load_freesolv(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the FreeSolv (Free Solvation Database) dataset.

    The task is to predict hydration free energy of small molecules in water [1]_ [2]_.
    Targets are in kcal/mol.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   642
    Recommended split          scaffold
    Recommended metric             RMSE
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
    .. [1] `Mobley, D.L., Guthrie, J.P.
        "FreeSolv: a database of experimental and calculated hydration free energies,
        with input files"
        J Comput Aided Mol Des 28, 711-720 (2014)
        <https://link.springer.com/article/10.1007/s10822-014-9747-x>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_freesolv
    >>> dataset = load_freesolv()
    >>> dataset  # doctest: +SKIP
    (['CN(C)C(=O)c1ccc(cc1)OC', ..., 'C1COCCO1'], array([-1.101e+01, -4.870e+00,  ..., 2.900e-01, -5.060e+00]))

    >>> dataset = load_freesolv(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                       SMILES  label
    0  CN(C)C(=O)c1ccc(cc1)OC -11.01
    1            CS(=O)(=O)Cl  -4.87
    2                CC(C)C=C   1.83
    3              CCc1cnccn1  -5.45
    4                CCCCCCCO  -4.21


    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_FreeSolv",
        filename="freesolv.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
