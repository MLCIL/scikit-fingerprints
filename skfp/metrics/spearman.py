import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.utils.functions import _get_sklearn_version


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "alternative": [StrOptions({"two-sided", "less", "greater"})],
        "return_p_value": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def spearman_correlation(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    *,
    alternative: str = "two-sided",
    return_p_value: bool = False,
) -> float:
    """
    Spearman correlation.

    Calculates Spearman's rank correlation coefficient (rho). It is a nonparametric
    measure of rank correlation. High value means that values of two variables change
    with a monotonic relationship.

    For identical inputs, i.e. exactly the same ``y_true`` and ``y_pred``, and when
    either vector is constant, NaN is returned, following Scipy behavior.

    Mainly intended for use as a quality metric, where a higher correlation between model
    prediction and ground truth is better. Can also be used for general correlation
    testing, by using ``alternative`` and ``return_p_value`` parameters.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis. By default, it checks if ground truth and estimated
        values differ in any direction.

    return_p_value : bool, default=False
        Whether to return p-value instead of correlation value.

    equal_values_result : float, default=1.0
        What value to return if ``y_true`` and ``y_pred`` are equal. This overrides the
        ``return_p_value`` parameter if necessary.

    Returns
    -------
    score : float
        Spearman correlation value, or p-value if ``return_p_value`` is True.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import spearman_correlation
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([3, 4, 5])
    >>> spearman_correlation(y_true, y_pred)  # doctest: +SKIP
    1.0
    >>> y_true = np.array([1, 2, 3, 4])
    >>> y_pred = np.array([4, 3, 2, 1])
    >>> spearman_correlation(y_true, y_pred)  # doctest: +SKIP
    -1.0
    """
    if _get_sklearn_version() < (1, 7):
        y_data_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput=None
        )
    else:
        y_data_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
            y_true, y_pred, sample_weight=None, multioutput=None
        )

    if np.all(np.isclose(y_true, y_pred)):
        return np.nan

    result = spearmanr(y_true, y_pred, alternative=alternative)
    return result.pvalue if return_p_value else result.statistic
