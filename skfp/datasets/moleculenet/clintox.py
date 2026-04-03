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
def load_clintox(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    r"""
    Load the ClinTox dataset.

    The task is to predict drug approval viability, by predicting clinical trial
    toxicity and final FDA approval status [1]_. Both tasks are binary.

    ==================   ========================
    Tasks                                       2
    Task type            multitask classification
    Total samples                            1477
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 2 label columns,
        FDA approval and clinical trial toxicity. Otherwise, returns SMILES as list
        of strings, and labels as a NumPy array (2D integer array).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 2 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_clintox
    >>> dataset = load_clintox()
    >>> dataset  # doctest: +SKIP
    (['[C@@H]1([C@@H]([C@@H]([C@H]([C@@H]([C@@H]1Cl)Cl)Cl)Cl)Cl)Cl', ..., 'S=[Se]=S'], array([[1, 0],
       [1, 0],
       [1, 0],
       ...,
       [1, 0],
       [1, 0],
       [1, 0]]))

    >>> dataset = load_clintox(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                      SMILES  FDA_APPROVED  CT_TOX
    0  [C@@H]1([C@@H]([C@@H]([C@H]([C@@H]([C@@H]1Cl)C...             1       0
    1  [C@H]([C@@H]([C@@H](C(=O)[O-])O)O)([C@H](C(=O)...             1       0
    2  [H]/[NH+]=C(/C1=CC(=O)/C(=C\C=c2ccc(=C([NH3+])...             1       0
    3  [H]/[NH+]=C(\N)/c1ccc(cc1)OCCCCCOc2ccc(cc2)/C(...             1       0
    4                                 [N+](=O)([O-])[O-]             1       0


    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ClinTox",
        filename="clintox.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
