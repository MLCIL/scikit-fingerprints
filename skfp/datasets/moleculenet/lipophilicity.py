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
def load_lipophilicity(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Lipophilicity dataset.

    The task is to predict octanol/water distribution coefficient (logD) at pH 7.4 [1]_.
    Targets are already log transformed, and are a unitless ratio.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  4200
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
    .. [1] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_lipophilicity
    >>> dataset = load_lipophilicity()
    >>> dataset  # doctest: +SKIP
    (['Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14', ..., 'CN1C(=O)C=C(CCc2ccc3ccccc3c2)N=C1N'],
        array([ 3.54, -1.18,  3.69, ...,  2.1 ,  2.65,  2.7 ]))

    >>> dataset = load_lipophilicity(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES  label
    0            Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14   3.54
    1  COc1cc(OC)c(cc1NC(=O)CSCC(=O)O)S(=O)(=O)N2C(C)...  -1.18
    2             COC(=O)[C@@H](N1CCc2sccc2C1)c3ccccc3Cl   3.69
    3  OC[C@H](O)CN1C(=O)C(Cc2ccccc12)NC(=O)c3cc4cc(C...   3.37
    4  Cc1cccc(C[C@H](NC(=O)c2cc(nn2C)C(C)(C)C)C(=O)N...   3.10



    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_Lipophilicity",
        filename="lipophilicity.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
