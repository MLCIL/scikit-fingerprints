from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdfiltercatalog import FilterCatalogParams
from sklearn.utils._param_validation import StrOptions

from skfp.bases.base_filter import BaseFilter
from skfp.bases.base_substructure_filter import BaseSubstructureFilter


class PAINSFilter(BaseSubstructureFilter):
    """
    Pan Assay Interference Compounds (PAINS) filter.

    Designed to filter out compounds that often give false positive results in
    high-throughput screens. Those are typically reactive molecules, which also
    contain functional groups resulting in high toxicity.

    There are 3 sets of PAINS filters, ordered by restrictiveness: A, B, C. PAINS A
    filters out the least molecules, while PAINS C is very aggressive. Rule definitions
    are available in the supplementary material of the original publication [1]_ (in
    Sybyl line notation), and in RDKit code [2]_ [3]_ [4]_ (as SMARTS patterns).

    Parameters
    ----------
    variant : {"all", "A", "B", "C"}, default="all"
        PAINS filter set. ``"all"`` means applying all three sets, commonly denoted
        as just "PAINS filters".

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
    .. [1] `Jonathan B. Baell and Georgina A. Holloway
        "New Substructure Filters for Removal of Pan Assay Interference Compounds
        (PAINS) from Screening Libraries and for Their Exclusion in Bioassays"
        J. Med. Chem. 2010, 53, 7, 2719-2740
        <https://doi.org/10.1021/jm901137j>`_

    .. [2] `RDKit PAINS A filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/pains_a.in>`_

    .. [3] `RDKit PAINS B filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/pains_b.in>`_

    .. [4] `RDKit PAINS C filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/pains_c.in>`_

    Examples
    --------
    >>> from skfp.filters import PAINSFilter
    >>> smiles = ["[C-]#N", "CC=O", "C1=CC(=O)C(=O)C=C1"]
    >>> filt = PAINSFilter()
    >>> filt
    PAINSFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['[C-]#N', 'CC=O']
    """

    _parameter_constraints = {
        **BaseFilter._parameter_constraints,
        "variant": [StrOptions({"all", "A", "B", "C"})],
    }

    def __init__(
        self,
        variant: str = "all",
        allow_one_violation: bool = False,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        self.variant = variant
        super().__init__(
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _load_filters(self) -> FilterCatalog:
        params = FilterCatalog.FilterCatalogParams()
        if self.variant == "all":
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        elif self.variant == "A":
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        elif self.variant == "B":
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        else:  # self.variant == "C"
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)

        filters = FilterCatalog.FilterCatalog(params)
        return filters
