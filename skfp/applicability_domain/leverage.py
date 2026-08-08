from numbers import Real

import numpy as np
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class LeverageADChecker(BaseADChecker):
    r"""
    Leverage method.

    Defines applicability domain based on the leverage statistic [1]_ [2]_ [3]_, which
    is a general distance of new point from the space of training data. Leverage is
    defined using the hat (projection / influence) matrix, and its formula is:

    .. math::

        leverage(x_i) = x^T_i (X^T X)^{-1} x_i

    This way, the new molecule is projected orthogonally into the space spanned
    by the training data, taking into consideration feature correlations via
    Gram matrix :math:`X^T X`. The leverage itself is used as an outlier score,
    with larger values meaning points further from the training data. Typical
    threshold is :math:`3(d+1)/n`,
    where `d` is the number of features and `n` is the number of training molecules.

    Typically, physicochemical properties (continuous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``. Features
    should not be too strongly correlated, as this can result in near-singular matrix
    that is not invertible.

    This method scales relatively well with number of samples, but not the number of
    features, as it builds :math:`d \times d` Gram matrix.

    Parameters
    ----------
    threshold: float or "auto", default="auto"
        Maximal leverage allowed for applicability domain. New points with larger
        leverage, i.e. distance to training set, are assumed to lie outside AD.

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
    .. [1] `Supratik Kar, Kunal Roy & Jerzy Leszczynski
        "Applicability Domain: A Step Toward Confident Predictions and Decidability for QSAR Modeling"
        In: Nicolotti, O. (eds) Computational Toxicology. Methods in Molecular Biology, vol 1800. Humana Press, New York, NY
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    .. [2] `Paola Gramatica
        "Principles of QSAR models validation: internal and external"
        QSAR & Combinatorial Science 26.5 (2007): 694-701
        <https://doi.org/10.1002/qsar.200610151>`_

    .. [3] `Leverage (statistics) on Wikipedia
        <https://en.wikipedia.org/wiki/Leverage_(statistics)>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import LeverageADChecker
    >>> X_train = np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 1.0]])
    >>> X_test = np.array([[1.0, 1.0], [1.0, 2.0], [20.0, 3.0]])
    >>> leverage_ad_checker = LeverageADChecker()
    >>> leverage_ad_checker
    LeverageADChecker()

    >>> leverage_ad_checker.fit(X_train)
    LeverageADChecker()

    >>> leverage_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "threshold": [Interval(Real, 0, None, closed="both"), StrOptions({"auto"})],
    }

    def __init__(
        self,
        threshold: float | str = "auto",
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

        # pseudoinverse, Gram matrix is singular for collinear features and when N < D
        self.inv_gram_ = np.linalg.pinv(X.T @ X)

        if self.threshold == "auto":
            n, d = X.shape
            self.threshold_ = 3 * (d + 1) / n
        else:
            self.threshold_ = self.threshold

        return self

    def predict(self, X: np.ndarray | csr_array) -> np.ndarray:  # noqa: D102
        leverages = self.score_samples(X)
        return leverages <= self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is equal to the
        leverage of each sample. Note that here lower score indicates sample
        more firmly inside AD.

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

        # leverages are the diagonal of the hat matrix X @ inv_gram_ @ X.T, but we
        # compute them row-wise to avoid materializing the n x n matrix
        leverages = np.sum((X @ self.inv_gram_) * X, axis=1)

        return leverages
