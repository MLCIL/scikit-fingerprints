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
def load_muv(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the MUV (Maximum Unbiased Validation) dataset.

    The task is to predict 17 targets designed for validation of virtual screening
    techniques, based on PubChem BioAssays. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                      17
    Task type            multitask classification
    Total samples                           93087
    Recommended split                    scaffold
    Recommended metric               AUPRC, AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 17 label columns,
        with names corresponding to MUV targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 17 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Ramsundar, Bharath, et al.
        "Massively multitask networks for drug discovery"
        arXiv:1502.02072 (2015)
        <https://arxiv.org/abs/1502.02072>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_muv
    >>> dataset = load_muv()
    >>> dataset  # doctest: +SKIP
    (['Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C', ..., 'COc1ccc([N+](=O)[O-])cc1NC(=O)c1ccc(C)o1'],
        array([[nan, nan, nan, ..., nan, nan, nan],
       [ 0.,  0., nan, ..., nan,  0.,  0.],
       [nan, nan,  0., ..., nan, nan,  0.],
       ...,
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ...,  0., nan, nan],
       [nan, nan, nan, ...,  0., nan, nan]]))

    >>> dataset = load_muv(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
                                                  SMILES  MUV-466  ...  MUV-858  MUV-859
    0    Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C      NaN  ...      NaN      NaN
    1                Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1      0.0  ...      0.0      0.0
    2  COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O...      NaN  ...      NaN      0.0
    3  O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1cc...      NaN  ...      0.0      NaN
    4                          NC(=O)NC(Cc1ccccc1)C(=O)O      0.0  ...      NaN      NaN
    ...

    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_MUV", filename="muv.csv", verbose=verbose
    )
    return df if as_frame else get_mol_strings_and_labels(df)
