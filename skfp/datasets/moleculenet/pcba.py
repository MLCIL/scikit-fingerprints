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
def load_pcba(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the PCBA (PubChem BioAssay) dataset.

    The task is to predict biological activity against 128 bioassays, generated
    by high-throughput screening (HTS). All tasks are binary active/non-active.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                     128
    Task type            multitask classification
    Total samples                          437929
    Recommended split                    scaffold
    Recommended metric               AUPRC, AUROC
    ==================   ========================

    **Warning:** in newer RDKit vesions, 2 molecules from the original dataset are
    not read correctly due to disallowed hypervalent states of some atoms
    (see [release notes](https://github.com/rdkit/rdkit/releases/tag/Release_2024_09_1)).
    This version of the PCBA dataset contains manual fixes for those molecules, removing
    additional hydrogens, e.g. ``[AlH3] -> [Al]``. In OGB scaffold split, used for
    benchmarking, both molecules are in the training set.

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 128 label columns,
        with names corresponding to biological activities (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 128 label columns
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
    >>> from skfp.datasets.moleculenet import load_pcba
    >>> dataset = load_pcba()
    >>> dataset  # doctest: +SKIP
    (['CC(=O)N1CCC2(CC1)NC(=O)N(c1ccccc1)N2', ..., 'CCN(CC(=O)Nc1ccc(C)c(S(=O)(=O)N(C)C)c1)Cc1ccccc1'], \
array([[ 0.,  0., nan, ..., nan, nan, nan],
       [ 0.,  0., nan, ..., nan, nan, nan],
       [nan,  0., nan, ..., nan, nan, nan],
       ...,
       [ 0.,  0.,  0., ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan]]))

    >>> dataset = load_pcba(as_frame=True)
    >>> dataset.head() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                                  SMILES  ...  PCBA-995
    0               CC(=O)N1CCC2(CC1)NC(=O)N(c1ccccc1)N2  ...       NaN
    1                         N#Cc1nnn(-c2ccc(Cl)cc2)c1N  ...       NaN
    2      COC(=O)c1ccc(NC(=O)c2ccccc2CC[N+](=O)[O-])cc1  ...       NaN
    3            CCC1NC(=O)c2cccnc2-n2c1nc1ccc(F)cc1c2=O  ...       NaN
    4  CC1=CC(=O)/C(=C2/C=C(C(=O)Nc3ccc(S(=O)(=O)Nc4o...  ...       NaN
    ...

    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_PCBA", filename="pcba.csv", verbose=verbose
    )
    return df if as_frame else get_mol_strings_and_labels(df)
