import numpy as np
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


class BeyondRo5Filter(BaseFilter):
    """
    Beyond Rule of Five (bRo5).

    Looser version of Lipinski's rule of 5, designed to cover novel orally bioavailable
    drugs that do not fulfill the original conditions [1]_. They are particularly
    suitable for "difficult" targets, allowing greater flexibility.

    Molecule can violate at most one of the rules (conditions):

    - molecular weight <= 1000
    - logP in range [-2, 10]
    - HBA <= 15
    - HBD <= 6
    - TPSA <= 250
    - number of rotatable bonds <= 20

    RDKit computes topological polar surface area (TPSA) in a specific way, differing
    slightly from published papers. See the documentation [2]_ for details.

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
    .. [1] `Bradley C. Doak, Jie Zheng, Doreen Dobritzsch and Jan Kihlberg
        "How Beyond Rule of 5 Drugs and Clinical Candidates Bind to Their Targets"
        J. Med. Chem. 2016, 59, 6, 2312-2327
        <https://pubs.acs.org/doi/10.1021/acs.jmedchem.5b01286>`_

    .. [2] `RDKit Implementation of the TPSA Descriptor
        <https://www.rdkit.org/docs/RDKit_Book.html#implementation-of-the-tpsa-descriptor>`_

    Examples
    --------
    >>> from skfp.filters import BeyondRo5Filter, LipinskiFilter
    >>> smiles = ["[C-]#N", "CC=O", "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C"]
    >>> filt_bro5 = BeyondRo5Filter()
    >>> filt_bro5
    BeyondRo5Filter()

    >>> filt_bro5.transform(smiles)
    ['[C-]#N', 'CC=O', 'O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C']

    >>> filt_ro5 = LipinskiFilter()
    >>> filt_ro5.transform(smiles)
    ['[C-]#N', 'CC=O']
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
            "MolWeight <= 1000",
            "-2 <= logP <= 10",
            "HBA <= 15",
            "HBD <= 6",
            "TPSA <= 250",
            "rotatable bonds <= 6",
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
            MolWt(mol) <= 1000,
            -2 <= MolLogP(mol) <= 10,
            CalcNumHBA(mol) <= 15,
            CalcNumHBD(mol) <= 6,
            CalcTPSA(mol) <= 250,
            CalcNumRotatableBonds(mol) <= 20,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
