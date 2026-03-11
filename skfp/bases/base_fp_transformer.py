from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np
import scipy.sparse
from joblib import effective_n_jobs
from rdkit.Chem import Mol
from rdkit.DataStructs import (
    IntSparseIntVect,
    LongSparseIntVect,
    SparseBitVect,
    UIntSparseIntVect,
    ULongSparseIntVect,
)
from scipy.sparse import csr_array, dok_array
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils._param_validation import InvalidParameterError
from tqdm import tqdm

from skfp.utils import run_in_parallel

"""
If you get MaybeEncodingError, first check any worker functions for exceptions!
That error isn't very informative, but gets thrown in Joblib multiprocessing.
"""

"""
Note that you need to do create RDKit objects *inside* the function that runs
in parallel, i.e. _calculate_fingerprint(), not in the constructor or outside in
general.
This is because Joblib needs to pickle data sent to workers, and RDKit objects
cannot be pickled, throwing TypeError: cannot pickle 'Boost.Python.function' object
"""


class BaseFingerprintTransformer(
    ABC, BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
):
    """
    Base class for fingerprints.

    Computes feature vectors - molecular fingerprints - from input molecules.

    This class is not meant to be used directly. If you want to use custom
    fingerprints, inherit from this class and implement the ``._calculate_fingerprint()``
    method. It operates on a minibatch of molecules for efficiency.

    For hashing (folding) fingerprints, the ``._hash_fingerprint_bits()`` may also be
    useful.

    Parameters
    ----------
    n_features_out : int
        Number of output features.

    requires_conformers : bool = False
        Whether the fingerprint requires 3D conformations as inputs.

    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.
    """

    # parameters common for all fingerprints
    _parameter_constraints: dict = {
        "count": ["boolean"],
        "sparse": ["boolean"],
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose", dict],
        "random_state": ["random_state", np.random.Generator],
    }

    def __init__(
        self,
        n_features_out: int,
        requires_conformers: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
    ):
        self.count = count
        self.sparse = sparse
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        # this, combined with ClassNamePrefixFeaturesOutMixin, automatically handles
        # set_output() API
        self._n_features_out = n_features_out
        self.n_features_out = self._n_features_out

        # indicate whether inputs need to be molecules with conformers computed and
        # conf_id integer property set; this allows programmatically checking which
        # fingerprints are 3D-based and require such input
        self.requires_conformers = requires_conformers

    def __sklearn_is_fitted__(self) -> bool:
        """
        Unused, kept for scikit-learn compatibility. This class assumes stateless
        transformers and always returns True.
        """
        return True

    def set_params(self, **params):  # noqa: D102
        super().set_params(**params)
        # for fingerprints that have both 2D and 3D versions, as indicated by use_3D
        # attribute, we need to also keep requires_conformers attribute in sync
        if hasattr(self, "use_3D"):
            self.requires_conformers = self.use_3D
        return self

    def fit(self, X: Sequence[str | Mol], y: Any | None = None, **fit_params):
        """
        Unused, kept for scikit-learn compatibility.

        Parameters
        ----------
        X : any
            Unused, kept for scikit-learn compatibility.

        y : any
            Unused, kept for scikit-learn compatibility.

        **fit_params : dict
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        self
        """
        self._validate_params()
        return self

    def fit_transform(self, X: Sequence[str | Mol], y: Any | None = None, **fit_params):
        """
        The same as ``.transform()`` method, kept for scikit-learn compatibility.

        Parameters
        ----------
        X : any
            See ``.transform()`` method.

        y : any
            See ``.transform()`` method.

        **fit_params : dict
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X_new : any
            See ``.transform()`` method.
        """
        return self.transform(X)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute fingerprints. Output shape depends on the inheriting class.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects. Depending on
            the implementation in the inheriting class, it may require using ``Mol``
            objects with computed conformations and with ``conf_id`` property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, any)
            Array with fingerprints.
        """
        self._validate_params()

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            if self.verbose:
                results = [self._calculate_fingerprint([mol]) for mol in tqdm(X)]
            else:
                results = self._calculate_fingerprint(X)
        else:
            results = run_in_parallel(
                self._calculate_fingerprint,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )

        if isinstance(results, (np.ndarray, csr_array)):
            return results
        else:
            return scipy.sparse.vstack(results) if self.sparse else np.vstack(results)

    @abstractmethod
    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        """
        Calculate fingerprints for a given input batch.

        :param X: subset of original X data
        :return: array containing calculated fingerprints for each molecule
        """
        raise NotImplementedError

    def _validate_params(self) -> None:
        # override scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None

    @staticmethod
    def _hash_fingerprint_bits(
        X: list[
            IntSparseIntVect
            | LongSparseIntVect
            | SparseBitVect
            | UIntSparseIntVect
            | ULongSparseIntVect
        ],
        fp_size: int,
        count: bool,
        sparse: bool,
    ) -> np.ndarray | csr_array:
        rdkit_vec_types = (
            IntSparseIntVect,
            LongSparseIntVect,
            SparseBitVect,
            UIntSparseIntVect,
            ULongSparseIntVect,
        )
        if not all(isinstance(x, rdkit_vec_types) for x in X):
            raise ValueError(
                f"Fingerprint hashing requires instances of one of: {rdkit_vec_types}"
            )

        shape = (len(X), fp_size)
        arr = (
            dok_array(shape, dtype=np.uint32)
            if sparse
            else np.zeros(shape, dtype=np.uint32)
        )

        if isinstance(X[0], SparseBitVect):
            for idx, x in enumerate(X):
                for fp_bit in x.GetOnBits():
                    arr[idx, fp_bit % fp_size] += 1
        else:
            for idx, x in enumerate(X):
                for fp_bit, val in x.GetNonzeroElements().items():
                    arr[idx, fp_bit % fp_size] += val

        arr = arr.tocsr() if sparse else arr
        arr = (arr > 0).astype(np.uint8) if not count else arr

        return arr
