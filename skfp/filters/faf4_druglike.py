import numpy as np
from rdkit.Chem import GetFormalCharge, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumHeteroatoms,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter

from .utils import (
    get_max_ring_size,
    get_non_carbon_to_carbon_ratio,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


class FAF4DruglikeFilter(BaseFilter):
    """
    FAFDrugs4 Drug-Like Soft filter.

    Designed as a part of FAFDrugs4 software [1]_ [2]_. Based on literature describing
    physico-chemical properties of drugs and their statistical analysis. Selected
    so that up to 90% of the 916 FDA-approved oral drugs fulfill the rules of this
    filter.

    Known to reject accepted, but atypical drugs, mostly due to high molecular weight
    (e.g. Rinfampin), hydrophobicity (e.g. Probucol), rotatable bonds (e.g. Aliskerin)
    or HBD (e.g. Kanamycin).

    Molecule must fulfill conditions:

    - molecular weight in range [100, 600]
    - logP in range [-3, 6]
    - HBA <= 12
    - HBD <= 7
    - TPSA <= 180
    - number of rotatable bonds <= 11
    - number of rigid bonds <= 30
    - number of rings <= 6
    - max ring size <= 18
    - number of carbons in range [3, 35]
    - number of heteroatoms in range [1, 15]
    - non-carbons to carbons ratio in range [0.1, 1.1]
    - number of charged functional groups <= 4
    - total formal charge in range [-4, 4]

    Note that the FAFDrugs4 uses ChemAxon for determining functional groups. We use
    their publicly available CXSMARTS list of functional groups [3]_. Phosphine and
    sulfoxide patterns could not be parsed by RDKit, so they were fixed appropriately.

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
    .. [1] `Details of physico-chemical property filters available in FAF-Drugs4
        <https://fafdrugs4.rpbs.univ-paris-diderot.fr/filters.html>`_

    .. [2] `D. Lagorce et al.
        "FAF-Drugs4: free ADME-tox filtering computations for chemical biology and
        early stages drug discovery"
        Bioinformatics, 33(22), 2017, 3658-3660
        <https://doi.org/10.1093/bioinformatics/btx491>`_

    .. [3] `ChemAxon documentation: Predefined Functional Groups and Named Molecule Groups
        <https://docs.chemaxon.com/display/docs/attachments/attachments_1829721_1_functionalgroups.cxsmi>`_

    Examples
    --------
    >>> from skfp.filters import FAF4DruglikeFilter
    >>> smiles = ["C", "CC(=O)Nc1ccc(O)cc1"]
    >>> filt = FAF4DruglikeFilter()
    >>> filt
    FAF4DruglikeFilter()

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
            "100 <= MolWeight <= 600",
            "-3 <= logP <= 6",
            "HBA <= 12",
            "HBD <= 7",
            "TPSA <= 180",
            "rotatable bonds <= 11",
            "rigid bonds <= 30",
            "rings <= 6",
            "max ring size <= 18",
            "3 <= carbon atoms <= 35",
            "1 <= heteroatoms <= 15",
            "0.1 <= non-carbon to carbon ratio <= 1.1",
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
            100 <= MolWt(mol) <= 600,
            -3 <= MolLogP(mol) <= 6,
            CalcNumHBA(mol) <= 12,
            CalcNumHBD(mol) <= 7,
            CalcTPSA(mol) <= 180,
            CalcNumRotatableBonds(mol) <= 11,
            get_num_rigid_bonds(mol) <= 30,
            CalcNumRings(mol) <= 6,
            get_max_ring_size(mol) <= 18,
            3 <= get_num_carbon_atoms(mol) <= 35,
            1 <= CalcNumHeteroatoms(mol) <= 15,
            0.1 <= get_non_carbon_to_carbon_ratio(mol) <= 1.1,
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
