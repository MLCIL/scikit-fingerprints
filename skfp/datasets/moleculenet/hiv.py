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
def load_hiv(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load the HIV dataset.

    The task is to predict ability of molecules to inhibit HIV replication [1]_ [2]_.

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                 41127
    Recommended split          scaffold
    Recommended metric            AUROC
    ==================   ==============

    **Warning:** in newer RDKit vesions, 7 molecules from the original dataset are
    not read correctly due to disallowed hypervalent states of some atoms
    (see [release notes](https://github.com/rdkit/rdkit/releases/tag/Release_2024_09_1)).
    This version of the HIV dataset contains manual fixes for those molecules, made
    by cross-referencing original NCI data, PubChem substructure search, and visualization
    with ChemAxon Marvin. In OGB scaffold split, used for benchmarking, first 2 of those
    problematic 7 are from the test set.

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default ``$HOME/scikit_learn_data``.

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
    .. [1] `AIDS Antiviral Screen Data
        <https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_hiv
    >>> dataset = load_hiv()
    >>> dataset  # doctest: +SKIP
    (['CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2', ..., \
'CCCCCC=C(c1cc(Cl)c(OC)c(-c2nc(C)no2)c1)c1cc(Cl)c(OC)c(-c2nc(C)no2)c1'], \
array([0, 0, 0, ..., 0, 0, 0]))

    >>> dataset = load_hiv(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                      SMILES  label
    0  CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)...      0
    1  C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3...      0
    2                   CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21      0
    3    Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1      0
    4                             O=S(=O)(O)CCS(=O)(=O)O      0


    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_HIV", filename="hiv.csv", verbose=verbose
    )
    return df if as_frame else get_mol_strings_and_labels(df)
