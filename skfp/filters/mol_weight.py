from numbers import Integral

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases.base_filter import BaseFilter


class MolecularWeightFilter(BaseFilter):
    """
    Molecular weight filter.

    Filters out molecules with mass in daltons outside the given range. Provided
    ``min_weight`` and ``max_weight`` are inclusive on both sides.

    Parameters
    ----------
    min_weight : int, default=0
        Minimal weight in daltons.

    max_weight : int, default=1000
        Maximal weight in daltons.

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

    Examples
    --------
    >>> from skfp.filters import MolecularWeightFilter
    >>> smiles = ["C", "O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    >>> filt = MolecularWeightFilter(max_weight=100)
    >>> filt
    MolecularWeightFilter(max_weight=100)

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C', 'O']
    """

    _parameter_constraints: dict = {
        **BaseFilter._parameter_constraints,
        "min_weight": [Interval(Integral, 0, None, closed="left")],
        "max_weight": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        min_weight: int = 0,
        max_weight: int = 1000,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        condition_names = [f"{min_weight} <= MolWeight <= {max_weight}"]
        super().__init__(
            condition_names=condition_names,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_weight < self.min_weight:
            raise InvalidParameterError(
                f"The max_weight parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_weight, got: "
                f"min_weight={self.min_weight}, max_weight={self.max_weight}"
            )

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        passed = self.min_weight <= MolWt(mol) <= self.max_weight

        if self.return_type == "condition_indicators":
            passed = np.array([passed], dtype=bool).reshape(-1, 1)

        return passed
