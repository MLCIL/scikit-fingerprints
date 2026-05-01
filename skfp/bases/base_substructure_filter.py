from abc import abstractmethod

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdfiltercatalog import FilterCatalog

from skfp.bases import BaseFilter


class BaseSubstructureFilter(BaseFilter):
    """
    Base class for substructure molecular filters.

    Substructure filters are defined using SMARTS patterns, which determine unwanted
    functional groups and other subgraphs.

    This class is not meant to be used directly. If you want to create custom
    filters, inherit from this class and override the ``._load_filters()``
    method. It registers and returns RDKit ``FilterCatalog``, see the RDKit
    documentation for details.

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
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        self._filters = self._load_filters()
        condition_names, condition_name_to_idx = self._load_condition_names_and_map()
        self._condition_name_to_idx = condition_name_to_idx
        super().__init__(
            condition_names=condition_names,
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    @abstractmethod
    def _load_filters(self) -> FilterCatalog:
        raise NotImplementedError

    def _load_condition_names_and_map(self) -> tuple[list[str], dict[str, int]]:
        condition_names = [
            self._filters.GetEntryWithIdx(i).GetDescription()
            for i in range(self._filters.GetNumEntries())
        ]
        condition_name_to_idx = {name: idx for idx, name in enumerate(condition_names)}
        return condition_names, condition_name_to_idx

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        matches = self._filters.GetMatches(mol)

        if self.return_type == "condition_indicators":
            matched_conditions_idxs = [
                self._condition_name_to_idx[match.GetDescription()] for match in matches
            ]
            result = np.zeros(len(self.condition_names), dtype=bool)
            result[matched_conditions_idxs] = True
            return result
        else:
            errors = len(matches)
            return not errors or (self.allow_one_violation and errors == 1)
