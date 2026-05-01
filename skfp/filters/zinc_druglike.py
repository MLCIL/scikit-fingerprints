import numpy as np
from rdkit.Chem import GetFormalCharge, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter
from skfp.filters.utils import (
    get_max_ring_size,
    get_non_carbon_to_carbon_ratio,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


class ZINCDruglikeFilter(BaseFilter):
    """
    ZINC druglike filter.

    Designed to keep only drug-like molecules [1]_. Based only on physico-chemical
    properties [2]_, since SMARTS for additional rules are not publicly available.
    See "filter_light.txt" section in the supplementary material of the original
    paper [1]_ for details.

    Molecule must fulfill conditions:

    - molecular weight in range [60, 600]
    - logP in range [-4, 6]
    - HBA <= 11
    - HBD <= 6
    - TPSA <= 150
    - number of rotatable bonds <= 12
    - number of rigid bonds <= 50
    - number of rings <= 7
    - max ring size <= 12
    - number of carbons >= 3
    - non-carbons to carbons ratio <= 2.0
    - number of charged functional groups <= 4
    - total formal charge in range [-4, 4]

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
    .. [1] `John J. Irwin, Brian K. Shoichet
        "ZINC − A Free Database of Commercially Available Compounds for Virtual Screening"
        J. Chem. Inf. Model. 2005, 45, 1, 177-182
        <https://doi.org/10.1021/ci049714+>`_

    .. [2] `Details of physico-chemical property filters available in FAF-Drugs4
        <https://fafdrugs4.rpbs.univ-paris-diderot.fr/filters.html>`_

    Examples
    --------
    >>> from skfp.filters import ZINCDruglikeFilter
    >>> smiles = ["C", "CC(=O)Nc1ccc(O)cc1"]
    >>> filt = ZINCDruglikeFilter()
    >>> filt
    ZINCDruglikeFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)Nc1ccc(O)cc1']
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
            "60 <= MolWeight <= 600",
            "-4 <= logP <= 6",
            "HBA <= 11",
            "HBD <= 6",
            "TPSA <= 150",
            "rotatable bonds <= 12",
            "rigid bonds <= 50",
            "rings <= 7",
            "max ring size <= 12",
            "carbon atoms >= 3",
            "non-carbon to carbon ratio <= 2.0",
            "charged functional groups <= 4",
            "-4 <= formal charge <= 4",
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
            60 <= MolWt(mol) <= 600,
            -4 <= MolLogP(mol) <= 6,
            CalcNumHBA(mol) <= 11,
            CalcNumHBD(mol) <= 6,
            CalcTPSA(mol) <= 150,
            CalcNumRotatableBonds(mol) <= 12,
            get_num_rigid_bonds(mol) <= 50,
            CalcNumRings(mol) <= 7,
            get_max_ring_size(mol) <= 12,
            get_num_carbon_atoms(mol) >= 3,
            get_non_carbon_to_carbon_ratio(mol) <= 2,
            get_num_charged_functional_groups(mol) <= 4,
            -4 <= GetFormalCharge(mol) <= 4,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
