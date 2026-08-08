from numbers import Integral, Real

import numpy as np
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class BoundingBoxADChecker(BaseADChecker):
    """
    Bounding box method.

    Defines applicability domain based on feature ranges in the training data.
    This creates a "bounding box" using their extreme values, and new molecules
    should lie in this distribution, i.e. have properties in the same ranges [1]_.

    Typically, physicochemical properties (continuous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``. This is
    particularly important if ``"three_sigma"`` is used as the percentile bound, as it
    assumes normal distribution.

    By default, the full range of training descriptors are allowed as AD. For stricter
    check, use ``percentile_lower`` and ``percentile_upper`` arguments to disallow
    extremely low or large values, respectively. For looser check, use ``num_allowed_violations``
    to allow a number of desrciptors to lie outside the given ranges.

    This method scales very well with both the number of samples and features.

    Parameters
    ----------
    percentile_lower : float or "three_sigma", default=0
        Lower bound of accepted feature value ranges. Float or integer value is interpreted as
        a percentile of descriptors in the training data for each feature. ``"three_sigma"``
        uses 3 standard deviations from the mean, a common rule-of-thumb for outliers
        assuming the normal distribution.

    percentile_upper : float or "three_sigma", default=100
        Upper bound of accepted feature value ranges. Float or integer value is interpreted as
        a percentile of descriptors in the training data for each feature. ``"three_sigma"``
        uses 3 standard deviations from the mean, a common rule-of-thumb for outliers
        assuming the normal distribution.

    num_allowed_violations : int, default=0
        Number of allowed violations of feature ranges. By default, all descriptors
        must lie inside the bounding box.

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
    .. [1] `Kar, S., Roy, K., Leszczynski, J.
        "Applicability Domain: A Step Toward Confident Predictions and Decidability
        for QSAR Modeling."
        Nicolotti, O. (eds) Computational Toxicology. Methods in Molecular Biology, vol 1800.
        Humana Press, New York, NY
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import BoundingBoxADChecker
    >>> X_train = np.array([[0.1, 0.2, 0.3], [1.0, 0.9, 0.8], [0.5, 0.5, 0.5]])
    >>> X_test = np.array([[0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.0, 0.9, 1.0]])
    >>> bb_ad_checker = BoundingBoxADChecker()
    >>> bb_ad_checker
    BoundingBoxADChecker()

    >>> bb_ad_checker.fit(X_train)
    BoundingBoxADChecker()

    >>> bb_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "percentile_lower": [Interval(Real, 0, 100, closed="both"), "three_sigma"],
        "percentile_upper": [Interval(Real, 0, 100, closed="both"), "three_sigma"],
        "num_allowed_violations": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        percentile_lower: float | str = 0,
        percentile_upper: float | str = 100,
        num_allowed_violations: int = 0,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.num_allowed_violations = num_allowed_violations

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)

        if (
            self.percentile_lower == "three_sigma"
            or self.percentile_upper == "three_sigma"
        ):
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

        if self.percentile_lower == "three_sigma":
            self.lower_bounds_ = mean - 3 * std
        else:
            self.lower_bounds_ = np.percentile(X, self.percentile_lower, axis=0)

        if self.percentile_upper == "three_sigma":
            self.upper_bounds_ = mean + 3 * std
        else:
            self.upper_bounds_ = np.percentile(X, self.percentile_upper, axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        outside_range = (self.lower_bounds_ > X) | (self.upper_bounds_ < X)
        violations = np.sum(outside_range, axis=1)
        passed = violations <= self.num_allowed_violations
        return passed

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is the number
        of feature ranges fulfilled by samples. It ranges between 0 and
        ``num_features``, where ``num_features`` means all descriptors are
        inside training data ranges.

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

        inside_range = (self.lower_bounds_ <= X) & (self.upper_bounds_ >= X)
        scores = np.sum(inside_range, axis=1)
        return scores
