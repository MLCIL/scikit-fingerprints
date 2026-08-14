from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class TOPKATADChecker(BaseADChecker):
    """
    TOPKAT (Optimal Prediction Space) method.

    Defines applicability domain using the Optimal Prediction Space (OPS) approach
    introduced in the TOPKAT system [1]_. The method transforms the input feature space
    into a normalized, centered and rotated space using PCA on a scaled [-1, 1] version
    of the training data (S-space). Each new sample is projected into the same OPS space,
    and a weighted distance (dOPS) from the center is computed.

    Samples are considered in-domain if their dOPS is below a threshold. By default,
    this threshold is computed as :math:`5 * D / (2 * N)`, where:
    - :math:``D`` is the number of input features,
    - :math:``N`` is the number of training samples.

    This method captures both the variance and correlation structure of the descriptors,
    and performs well for detecting global outliers in descriptor space.

    Parameters
    ----------
    threshold : float, default=None
        Optional user-defined threshold for dOPS. If provided, overrides the default
        analytical threshold :math:`5 * D / (2 * N)`. Lower values produce stricter domains.

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
    .. [1] Gombar, V. K.
       `"Method and apparatus for validation of model-based predictions."
       U.S. Patent No. 6,036,349. Washington, DC: U.S. Patent and Trademark Office.
       <https://patents.google.com/patent/US6036349A/en>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import TOPKATADChecker
    >>> from sklearn.datasets import make_blobs
    >>> X_train, _ = make_blobs(n_samples=100, centers=2, n_features=5, random_state=0)
    >>> X_test = X_train[:5]

    >>> topkat_ad_checker = TOPKATADChecker()
    >>> topkat_ad_checker.fit(X_train)
    TOPKATADChecker()

    >>> topkat_ad_checker.predict(X_test)
    array([ True,  True,  True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "threshold": [Interval(Real, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        threshold: float | None = None,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.threshold = threshold

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)

        self.X_min_ = X.min(axis=0)
        self.X_max_ = X.max(axis=0)
        self.range_ = self.X_max_ - self.X_min_

        if self.threshold:
            self.threshold_ = self.threshold
        else:
            self.threshold_ = (5 * X.shape[1]) / (2 * X.shape[0])

        # TOPKAT S-space: feature-wise scaling of X to [-1, 1]
        # avoid division by zero: where range==0, denom=1 => scaled value will be 0
        self.denom_ = np.where((self.range_) != 0, (self.range_), 1.0)
        S = (2 * X - self.X_max_ - self.X_min_) / self.denom_

        # augment with bias (1) so the subsequent rotation (PCA) includes the intercept
        S_bias = np.c_[np.ones(S.shape[0]), S]

        cov_matrix = S_bias.T @ S_bias
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        self.eigen_val_ = np.real(eigvals)
        self.eigen_vec_ = np.real(eigvecs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        dOPS = self._compute_dops(X)
        return dOPS < self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples.
        It is defined as the weighted distance (dOPS) from
        the center of the training data in the OPS-transformed space.
        Lower values indicate higher similarity to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Distance scores from the center of the Optimal Prediction Space (dOPS).
        """
        return self._compute_dops(X)

    def _compute_dops(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        # apply the same S-space transform as in fit().
        Ssample = (2 * X - self.X_max_ - self.X_min_) / self.denom_

        # add bias term to match the augmented space used to compute eigenvectors.
        Ssample_bias = np.c_[np.ones(Ssample.shape[0]), Ssample]

        # project to OPS space
        OPS_sample = Ssample_bias @ self.eigen_vec_

        inv_eigval = np.divide(
            1.0,
            self.eigen_val_,
            out=np.zeros_like(self.eigen_val_),
            where=self.eigen_val_ != 0,
        )
        dOPS = np.sum((OPS_sample**2) * inv_eigval, axis=1)
        return dOPS
