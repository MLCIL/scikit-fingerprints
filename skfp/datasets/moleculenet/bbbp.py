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
def load_bbbp(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the BBBP (Blood-Brain Barrier Penetration) dataset.

    The task is to predict blood-brain barrier penetration (barrier permeability)
    of small drug-like molecules [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                  2039
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
    .. [1] `Ines Filipa Martins et al.
        "A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling"
        J. Chem. Inf. Model. 2012, 52, 6, 1686-1697
        <https://pubs.acs.org/doi/10.1021/ci300124c>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_bbbp
    >>> dataset = load_bbbp()
    >>> dataset  # doctest: +SKIP
    (['[Cl].CC(C)NCC(O)COc1cccc2ccccc12', ..., '[N+](=NCC(=O)N[C@@H]([C@H](O)C1=CC=C([N+]([O-])=O)C=C1)CO)=[N-]'], \
array([1, 1, 1, ..., 1, 1, 1]))

    >>> dataset = load_bbbp(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                  SMILES  label
    0                   [Cl].CC(C)NCC(O)COc1cccc2ccccc12      1
    1           C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl      1
    2  c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO...      1
    3                   C1CCN(CC1)Cc1cccc(c1)OCCCNC(=O)C      1
    4  Cc1onc(c2ccccc2Cl)c1C(=O)N[C@H]3[C@H]4SC(C)(C)...      1

    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_BBBP", filename="bbbp.csv", verbose=verbose
    )
    return df if as_frame else get_mol_strings_and_labels(df)
