import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt

from skfp.bases.base_filter import BaseFilter


class GSKFilter(BaseFilter):
    """
    GSK rule (4/400) filter.

    Compute GSK Rule (4/400) for druglikeness using interpretable ADMET rule of thumb [1]_.

    Molecule must fulfill conditions:

    - molecular weight <= 400
    - logP <= 4

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
    .. [1] `Glesson, M. P.
        "Generation of a Set of Simple, Interpretable ADMET Rules of Thumb"
        J. Med. Chem. 2008, 51, 4, 817-834
        <https://pubs.acs.org/doi/10.1021/jm701122q>`_

    Examples
    --------
    >>> from skfp.filters import GSKFilter
    >>> smiles = ["C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C"]
    >>> filt = GSKFilter()
    >>> filt
    GSKFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int = 0,
    ):
        condition_names = [
            "MolWeight <= 400",
            "logP <= 4",
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
        rules = [
            MolWt(mol) <= 400,
            MolLogP(mol) <= 4,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
