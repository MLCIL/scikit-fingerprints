from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter


class RuleOfThreeFilter(BaseFilter):
    """
    Rule of three (Ro3).

    Rule optimized to search for fragment-based lead-like compounds with desired properties.
    It was described in [1]_.

    Molecule must fulfill conditions:

    - molecular weight <= 300
    - HBA <= 3
    - HBD <= 3
    - logP <= 3

    Additionally, an extended version of this rule has been proposed, which adds two conditions:

    - number of rotatable bonds <= 3
    - TPSA <= 60

    Parameters
    ----------
    extended : bool, default=False
        Whether to use an extended version of this rule, additionally including TPSA and
        rotatable bonds conditions.

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
    .. [1] `Congreve, M., Carr, R., Murray, C., & Jhoti, H.
        "A 'Rule of Three' for fragment-based lead discovery?"
        Drug Discovery Today, 8(19), 876-877.
        <https://doi.org/10.1016/S1359-6446(03)02831-9>`_

    Examples
    --------
    >>> from skfp.filters import RuleOfThreeFilter
    >>> smiles = ['C=CCNC(=S)NCc1ccccc1OC', 'C=CCOc1ccc(Br)cc1/C=N/O', 'C=CCNc1ncnc2ccccc12']
    >>> filt = RuleOfThreeFilter()
    >>> filt
    RuleOfThreeFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C=CCNC(=S)NCc1ccccc1OC', 'C=CCOc1ccc(Br)cc1/C=N/O', 'C=CCNc1ncnc2ccccc12']
    >>> filt = RuleOfThreeFilter(extended=True)
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C=CCNc1ncnc2ccccc12']
    """

    def __init__(
        self,
        extended: bool = False,
        allow_one_violation: bool = False,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        condition_names = [
            "MolWeight <= 300",
            "HBA <= 3",
            "HBD <= 3",
            "logP <= 3",
        ]
        if extended:
            condition_names += [
                "rotatable bonds <= 3",
                "TPSA <= 60",
            ]
        super().__init__(
            condition_names=condition_names,
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.extended = extended

    def _apply_mol_filter(self, mol: Mol) -> bool | list[bool]:
        rules = [
            MolWt(mol) <= 300,
            CalcNumHBA(mol) <= 3,
            CalcNumHBD(mol) <= 3,
            MolLogP(mol) <= 3,
        ]
        if self.extended:
            rules += [CalcNumRotatableBonds(mol) <= 3, CalcTPSA(mol) <= 60]

        if self.return_type == "condition_indicators":
            return rules

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
