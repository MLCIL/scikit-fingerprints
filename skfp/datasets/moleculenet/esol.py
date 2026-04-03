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
def load_esol(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the ESOL (Estimated SOLubility) dataset.

    The task is to predict aqueous solubility [1]_ [2]_. Targets are log-transformed,
    and the unit is log mols per litre (log Mol/L).

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1128
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
    .. [1] `John S. Delaney
        "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure"
        J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000-1005
        <https://pubs.acs.org/doi/10.1021/ci034243x>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_esol
    >>> dataset = load_esol()
    >>> dataset  # doctest: +SKIP
    (['OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', ..., 'COP(=O)(OC)OC(=CCl)c1cc(Cl)c(Cl)cc1Cl'],
        array([-0.77 , -3.3  , -2.06 , ..., -3.091, -3.18 , -4.522]))

    >>> dataset = load_esol(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES  label
    0  OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)...  -0.77
    1                             Cc1occc1C(=O)Nc2ccccc2  -3.30
    2                               CC(C)=CCCC(C)=CC(=O)  -2.06
    3                 c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43  -7.87
    4                                            c1ccsc1  -1.33


    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ESOL",
        filename="esol.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
