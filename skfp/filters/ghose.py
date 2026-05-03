import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcCrippenDescriptors, CalcNumAtoms

from skfp.bases.base_filter import BaseFilter


class GhoseFilter(BaseFilter):
    """
    Ghose Filter.

    Used to searching for drug-like molecules [1]_.

    Molecule must fulfill conditions:

    - molecular weight in range [160, 400]
    - logP in range [-0.4, 5.6]
    - number of atoms in range [20, 70]
    - molar refractivity in range [40, 130]

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

    return_type : {"mol", "indicators", "condition_indicators"}, default="mol"
        What values to return as the filtering result.

        - ``"mol"`` - return a list of molecules remaining in the dataset after filtering
        - ``"indicators"`` - return a binary vector with indicators which molecules pass
          the filter (1) and which would be removed (0)
        - ``"condition_indicators"`` - return a Pandas DataFrame with molecules in rows,
          filter conditions in columns, and 0/1 indicators whether a given condition was
          fulfilled by a given molecule

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Ghose, A. K., Viswanadhan, V. N., & Wendoloski, J. J.
        "A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery. 1.
        A Qualitative and Quantitative Characterization of Known Drug Databases."
        Journal of Combinatorial Chemistry, 1(1), 55-68.
        <https://doi.org/10.1021/cc9800071>`_

    Examples
    --------
    >>> from skfp.filters import GhoseFilter
    >>> smiles = ["CC(=O)C1=C(O)C(=O)N(CCc2c[nH]c3ccccc23)C1c1ccc(C)cc1", "CC(=O)c1c(C)n(CC2CCCO2)c2ccc(O)cc12",\
    "CC(=O)c1c(C(C)=O)c(C)n(CCCCn2c(C)c(C(C)=O)c(C(C)=O)c2C)c1C"]
    >>> filt = GhoseFilter()
    >>> filt
    GhoseFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)C1=C(O)C(=O)N(CCc2c[nH]c3ccccc23)C1c1ccc(C)cc1', 'CC(=O)c1c(C)n(CC2CCCO2)c2ccc(O)cc12']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        condition_names = [
            "160 <= MolWeight <= 400",
            "-0.4 <= logP <= 5.6",
            "20 <= atoms <= 70",
            "40 <= molar refractivity <= 130",
        ]
        super().__init__(
            condition_names=condition_names,
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        logp, mr = CalcCrippenDescriptors(mol)
        rules = [
            160 <= MolWt(mol) <= 400,
            -0.4 <= logp <= 5.6,
            20 <= CalcNumAtoms(mol) <= 70,
            40 <= mr <= 130,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
