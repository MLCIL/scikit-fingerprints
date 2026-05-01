import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcNumRotatableBonds

from skfp.bases.base_filter import BaseFilter


class TiceInsecticidesFilter(BaseFilter):
    r"""
    Tice rule for insecticides.

    Rule established based on statistical analysis of insecticides molecules [1]_.
    Designed specifically for insecticides, not general pesticides or other agrochemicals.

    Molecule must fulfill conditions:

    - molecular weight in range [150, 500]
    - logP in range [0, 5]
    - HBD <= 2
    - HBA in range [1, 8]
    - number of rotatable bonds <= 11

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
    .. [1] `Tice, C.M.,
        "Selecting the right compounds for screening:
        does Lipinski's Rule of 5 for pharmaceuticals apply to agrochemicals?"
        Pest. Manag. Sci., 57: 3-16.
        <https://doi.org/10.1002/1526-4998(200101)57:1\<3::AID-PS269\>3.0.CO;2-6>`_

    Examples
    --------
    >>> from skfp.filters import TiceInsecticidesFilter
    >>> smiles = ["O=C(CC1COc2ccccc2O1)NCCc1ccccc1", "O=C(Nc1cccc(Cl)c1)N1CCCC1", "CNc1nc(N)c([N+](=O)[O-])c(NCCO)n1"]
    >>> filt = TiceInsecticidesFilter()
    >>> filt
    TiceInsecticidesFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['O=C(CC1COc2ccccc2O1)NCCc1ccccc1', 'O=C(Nc1cccc(Cl)c1)N1CCCC1']
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
            "150 <= MolWeight <= 500",
            "0 <= logP <= 5",
            "HBD <= 2",
            "1 <= HBA <= 8",
            "rotatable bonds <= 11",
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
            150 <= MolWt(mol) <= 500,
            0 <= MolLogP(mol) <= 5,
            CalcNumHBD(mol) <= 2,
            1 <= CalcNumHBA(mol) <= 8,
            CalcNumRotatableBonds(mol) <= 11,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
