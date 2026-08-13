from collections.abc import Callable
from numbers import Integral, Real

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker
from skfp.distances import (
    _BULK_METRIC_NAMES as SKFP_BULK_METRIC_NAMES,
)
from skfp.distances import (
    _BULK_METRICS as SKFP_BULK_METRICS,
)
from skfp.distances import (
    _METRIC_NAMES as SKFP_METRIC_NAMES,
)
from skfp.distances import (
    _METRICS as SKFP_METRICS,
)

METRIC_FUNCTIONS = {**SKFP_METRICS, **SKFP_BULK_METRICS}
METRIC_NAMES = set(SKFP_METRIC_NAMES) | set(SKFP_BULK_METRIC_NAMES)


class KNNADChecker(BaseADChecker):
    r"""
    k-nearest neighbors applicability domain checker.

    This method determines whether a query molecule falls within the applicability
    domain by comparing its distance to k nearest neighbors [1]_ [2]_ [3]_ in the training set,
    using a threshold derived from the training data.

    The applicability domain is defined as one of:

    - the mean distance to k nearest neighbors,
    - the distance to k-th nearest neighbor (max distance),
    - the distance to the closest neighbor from the training set (min distance)

    A threshold is then set at the 95th percentile of these aggregated distances.
    Query molecules with an aggregated distance to their k nearest neighbors below
    this threshold are considered within the applicability domain.

    This implementation supports binary and count Tanimoto similarity metrics.

    Parameters
    ----------
    k : int, default=1
        Number of nearest neighbors to consider for distance calculations.
        Must be smaller than the number of training samples.

    metric: Callable or string, default="tanimoto_binary_distance"
        Distance metric to use.

    agg: "mean" or "max" or "min", default="mean"
        Aggregation method for distances to k nearest neighbors:

        - "mean": average distance
        - "max": maximum distance, to k-th neighbor
        - "min": minimal distance, to the nearest neighbor

    threshold : float, default=95
        Percentile of distance distribution, used as the threshold for determining the
        applicability domain. Value in range ``[0, 100]``.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Klingspohn, W., Mathea, M., ter Laak, A. et al.
        "Efficiency of different measures for defining the applicability domain of
        classification models."
        J Cheminform 9, 44 (2017)
        <https://doi.org/10.1186/s13321-017-0230-2>`_

    .. [2] `Harmeling, S., Dornhege G., Tax D., Meinecke F., Müller KR.
        "From outliers to prototypes: Ordering data."
        Neurocomputing, 69, 13, pages 1608-1618, (2006)
        <https://doi.org/10.1016/j.neucom.2005.05.015>`_

    .. [3] `Kar S., Roy K., Leszczynski J.
        "Applicability Domain: A Step Toward Confident Predictions and Decidability for QSAR Modeling"
        Methods Mol Biol, 1800, pages 141-169, (2018)
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    Examples
    --------
    >>> from skfp.applicability_domain import KNNADChecker
    >>> import numpy as np
    >>> X_train_binary = np.array([
    ...     [1, 1, 1],
    ...     [0, 1, 1],
    ...     [0, 0, 1]
    ... ])
    >>> X_test_binary = 1 - X_train_binary
    >>> knn_ad_checker_binary = KNNADChecker(k=2, metric="tanimoto_binary_distance", agg="mean")
    >>> knn_ad_checker_binary.fit(X_train_binary)
    KNNADChecker(k=2)

    >>> knn_ad_checker_binary.predict(X_test_binary)
    array([False, False,  True])

    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "k": [Interval(Integral, 1, None, closed="left")],
        "metric": [callable, StrOptions(METRIC_NAMES)],
        "agg": [StrOptions({"mean", "max", "min"})],
        "threshold": [None, Interval(Real, 0, 100, closed="both")],
    }

    def __init__(
        self,
        k: int = 1,
        metric: str | Callable = "tanimoto_binary_distance",
        agg: str = "mean",
        threshold: float = 95,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.k = k
        self.metric = metric
        self.agg = agg
        self.threshold = threshold

    def _validate_params(self) -> None:
        super()._validate_params()
        if isinstance(self.metric, str) and self.metric not in METRIC_FUNCTIONS:
            raise InvalidParameterError(
                f"Allowed metrics: {METRIC_NAMES}. Got: {self.metric}"
            )

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)
        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) must be smaller than or equal to the number of training samples ({X.shape[0]})"
            )

        self.X_train_ = X

        # number of neighbors aggregated when scoring a sample; for "min" only the
        # single nearest neighbor matters, regardless of k
        self._k_used = 1 if self.agg == "min" else self.k

        if self.agg == "min" and self.k > 1:
            raise UserWarning(
                f"k should be must be 1 when agg='min', because only "
                f"the single nearest neighbor matters, got {self.k}"
            )

        if callable(self.metric):
            metric_func = self.metric
        elif isinstance(self.metric, str) and self.metric in METRIC_FUNCTIONS:
            metric_func = METRIC_FUNCTIONS[self.metric]
        else:
            raise KeyError(
                f"Unknown metric: {self.metric}. Must be a callable or one of {list(METRIC_FUNCTIONS.keys())}"
            )

        self.knn_ = NearestNeighbors(
            n_neighbors=self._k_used, metric=metric_func, n_jobs=self.n_jobs
        )
        self.knn_.fit(X)

        # each training point is its own nearest neighbor, so we ask for one neighbor
        # more and drop it, to calibrate the threshold on the same number of
        # neighbors that .score_samples() aggregates for new molecules
        k_nearest, _ = self.knn_.kneighbors(
            X, n_neighbors=min(len(X), self._k_used + 1)
        )
        k_nearest = k_nearest[:, 1:]

        agg_dists = self._get_agg_dists(k_nearest)
        self.threshold_ = np.percentile(agg_dists, self.threshold)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        return self.score_samples(X) <= self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is the distance to
        the ``k`` nearest training molecules, aggregated as specified by ``agg``.
        Note that here lower score indicates sample more firmly inside AD.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Applicability domain scores of samples.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        k_nearest, _ = self.knn_.kneighbors(X, n_neighbors=self._k_used)
        return self._get_agg_dists(k_nearest)

    def _get_agg_dists(self, k_nearest: np.ndarray) -> np.ndarray:
        if self.agg == "mean":
            agg_dists = np.mean(k_nearest, axis=1)
        elif self.agg == "max":
            agg_dists = np.max(k_nearest, axis=1)
        else:  # "min"
            agg_dists = np.min(k_nearest, axis=1)

        return agg_dists
