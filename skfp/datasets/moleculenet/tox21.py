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
def load_tox21(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the Tox21 dataset.

    The task is to predict 12 toxicity targets, including nuclear receptors and
    stress response pathways. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                      12
    Task type            multitask classification
    Total samples                            7831
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    **Warning:** in newer RDKit vesions, 8 molecules from the original dataset
    are not read correctly due to disallowed hypervalent states of their aluminium
    atoms (see [release notes](https://github.com/rdkit/rdkit/releases/tag/Release_2024_09_1)).
    This version of the Tox21 dataset contains manual fixes for those molecules,
    removing additional hydrogens, e.g. ``[AlH3] -> [Al]``. In OGB scaffold split, used
    for benchmarking, only the first 1 of those problematic 8 is from the test set.

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 12 label columns,
        with names corresponding to toxicity targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 12 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Tox21 Challenge
        <https://tripod.nih.gov/tox21/challenge/>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_tox21
    >>> dataset = load_tox21()
    >>> dataset  # doctest: +SKIP
    (['CCOc1ccc2nc(S(N)(=O)=O)sc2c1', ..., 'COc1ccc2c(c1OC)CN1CCc3cc4c(cc3C1C2)OCO4'], \
array([[ 0.,  0.,  1., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ..., nan,  0.,  0.],
           [nan, nan, nan, ...,  0., nan, nan],
           ...,
           [ 1.,  1.,  0., ...,  0.,  0.,  0.],
           [ 1.,  1.,  0., ...,  0.,  1.,  1.],
           [ 0.,  0., nan, ...,  0.,  1.,  0.]]))



    >>> dataset = load_tox21(as_frame=True)
    >>> dataset.head() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                                  SMILES  NR-AR  ...  SR-MMP  SR-p53
    0                       CCOc1ccc2nc(S(N)(=O)=O)sc2c1    0.0  ...     0.0     0.0
    1                          CCN1C(=O)NC(c2ccccc2)C1=O    0.0  ...     0.0     0.0
    2  CC[C@]1(O)CC[C@H]2[C@@H]3CCC4=CCCC[C@@H]4[C@H]...    NaN  ...     NaN     NaN
    3                    CCCN(CC)C(CC)C(=O)Nc1c(C)cccc1C    0.0  ...     0.0     0.0
    4                          CC(O)(P(=O)(O)O)P(=O)(O)O    0.0  ...     0.0     0.0
    ...

    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_Tox21",
        filename="tox21.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
