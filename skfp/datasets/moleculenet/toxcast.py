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
def load_toxcast(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the ToxCast dataset.

    The task is to predict 617 toxicity targets from a large library of compounds
    based on in vitro high-throughput screening. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                     617
    Task type            multitask classification
    Total samples                            8576
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 617 label columns,
        with names corresponding to toxicity targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 617 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Ann M. Richard et al.
        "ToxCast Chemical Landscape: Paving the Road to 21st Century Toxicology"
        Chem. Res. Toxicol. 2016, 29, 8, 1225-1251
        <https://pubs.acs.org/doi/10.1021/acs.chemrestox.6b00135>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_toxcast
    >>> dataset = load_toxcast()
    >>> dataset  # doctest: +SKIP
    (['[O-][N+](=O)C1=CC=C(Cl)C=C1', ..., 'CN1CC2=C(N[C@H](CC(O)=O)C1=O)C=CC(=C2)C(=O)N1CCC(CC1)C1CCNCC1'], \
array([[ 0.,  0., nan, ...,  0.,  0.,  0.],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]]))

    >>> dataset = load_toxcast(as_frame=True)
    >>> dataset.head() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                          SMILES  ...  Tanguay_ZF_120hpf_YSE_up
    0                [O-][N+](=O)C1=CC=C(Cl)C=C1  ...                       0.0
    1  C[SiH](C)O[Si](C)(C)O[Si](C)(C)O[SiH](C)C  ...                       NaN
    2                   CN1CCN(CC1)C(=O)C1CCCCC1  ...                       NaN
    3                 NC1=CC=C(C=C1)[N+]([O-])=O  ...                       0.0
    4                 OC1=CC=C(C=C1)[N+]([O-])=O  ...                       0.0
    ...

    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ToxCast",
        filename="toxcast.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
