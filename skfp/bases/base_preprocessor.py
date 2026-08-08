import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral

from joblib import effective_n_jobs
from rdkit.Chem import Mol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import InvalidParameterError
from tqdm import tqdm

from skfp.utils import run_in_parallel
from skfp.utils.parallel import get_tqdm_settings


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for preprocessing molecules.

    This is a generic class for various preprocessing operations. It is not meant
    to be used directly. If you want to create custom preprocessing steps, inherit
    from this class and override the ``._transform_batch()`` method. It gets a
    minibatch of molecules and outputs the preprocessed results, depending on the
    implementation, e.g. molecules, vectors, or boolean indicators.

    Parameters
    ----------
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    suppress_warnings: bool, default=False
        Whether to suppress warnings and errors during processing operations.

    verbose : int, default=0
        Controls the verbosity when processing molecules.
    """

    # parameters common for all fingerprints
    _parameter_constraints: dict = {
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "suppress_warnings": ["boolean"],
        "verbose": ["verbose", dict],
    }

    def __init__(
        self,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        suppress_warnings: bool = False,
        verbose: int | dict = 0,
    ):
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.suppress_warnings = suppress_warnings
        self.verbose = verbose

    def __sklearn_is_fitted__(self) -> bool:
        """
        Unused, kept for scikit-learn compatibility. This class assumes stateless
        transformers and always returns True.
        """
        return True

    def fit(self, X, y=None, **fit_params):
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

    def fit_transform(self, X, y=None, **fit_params):
        """
        The same as ``.transform()`` method, kept for scikit-learn compatibility.

        Parameters
        ----------
        X : any
            See ``.transform()`` method.

        y : any
            Unused, kept for scikit-learn compatibility.

        **fit_params : dict
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X_new : any
            See ``.transform()`` method.
        """
        return self.transform(X)

    def transform(self, X: Sequence[str | Mol], copy: bool = False):
        """
        Transform inputs. Output type depends on the inheriting class, but should
        be a sequence with the same length as input.

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
        X : {sequence, array-like} of shape (n_samples, any)
            Transformed inputs.
        """
        self._validate_params()

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            tqdm_settings = get_tqdm_settings(self.verbose, total=len(X))
            if not tqdm_settings["disable"]:
                # one molecule at a time, so that the progress bar can advance
                results = [
                    self._transform_batch([mol]) for mol in tqdm(X, **tqdm_settings)
                ]
                results = list(itertools.chain.from_iterable(results))
            else:
                results = self._transform_batch(X)
        else:
            results = run_in_parallel(
                self._transform_batch,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                flatten_results=True,
                verbose=self.verbose,
            )

        return results

    @abstractmethod
    def _transform_batch(self, X):
        raise NotImplementedError

    def _validate_params(self) -> None:
        # override scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
