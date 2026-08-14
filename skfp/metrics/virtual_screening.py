from numbers import Real
from operator import itemgetter

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcRIE
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.multiclass import type_of_target


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "fraction": [Interval(Real, 0, 1, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def enrichment_factor(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    fraction: float = 0.05,
) -> float:
    r"""
    Enrichment factor (EF).

    EF at fraction ``X`` is calculated as the number of actives found by the model,
    divided by the expected number of actives from a random ranking. See also [1]_
    for details.

    We have ``n`` actives, ``N`` test molecules, and percentage ``fraction`` of the
    top molecules (e.g. 0.05). Take ``fraction * N`` compounds with highest ``y_score``
    values, and mark the number of actives among them as ``a``. Random classifier
    would get on average ``fraction * n`` actives. Enrichment factor
    EF(X) is then defined as a ratio:

    .. math::

        EF(X) = \frac{a}{X*n}

    Minimal value is 0. Maximal value depends on the fraction of actives in the
    dataset, and is equal to ``1/X`` if ``X >= n/N``, and ``N/n`` otherwise. Model
    as good as random guessing would get 1. Note that values depend on the ratio
    of actives in the dataset and ``fraction`` value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Target scores, e.g. probability of the positive class, or similarities to
        active compounds.

    fraction : float, default=0.05
        Fraction of the dataset used for calculating the enrichment. Common values
        are 0.01 and 0.05. Note that this value affects the possible value range.

    Returns
    -------
    score : float
        Enrichment factor value.

    References
    ----------
    .. [1] `Riniker, S., Landrum, G.A.
        "Open-source platform to benchmark fingerprints for ligand-based virtual screening"
        J Cheminform 5, 26 (2013)
        <https://doi.org/10.1186/1758-2946-5-26>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import enrichment_factor
    >>> y_true = [0, 0, 1]
    >>> y_score = [0.1, 0.2, 0.7]
    >>> enrichment_factor(y_true, y_score)
    3.0
    """
    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError(
            f"Enrichment factor is only defined for binary y_true, got {y_type}"
        )

    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    # sort by score only, so that ties keep the original input order; sorting the
    # (score, label) tuples would rank actives above inactives on equal scores,
    # optimistically inflating the metric
    scores = sorted(zip(y_score, y_true, strict=True), key=itemgetter(0), reverse=True)

    # RDKit returns a list, we have to extract the actual float
    # it can sometimes return an empty list, so we catch that and return 0.0 then, see:
    # https://github.com/rdkit/rdkit/issues/7981
    ef = CalcEnrichment(scores, col=1, fractions=[fraction])
    return 0.0 if not len(ef) else ef[0]


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "alpha": [Interval(Real, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def rie_score(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    alpha: float = 20,
) -> float:
    r"""
    Robust Initial Enhancement (RIE).

    Exponentially weighted alternative to enrichment factor (EF), aimed to fix the
    problem of high variance with small number of actives. The exponential weight
    ``alpha`` is roughly equivalent to ``1 / fraction`` from EF. See [1]_ [2]_ [3]_
    for details.

    We have ``n`` actives, ``N`` test molecules, weight ``alpha``, and ``r_i`` is
    the rank of the i-th active from the test set. Then, RIE is defined as:

    .. math::

        RIE(\alpha) = \frac{N}{n}
            \frac{\sum_{i=1}^n e^{-\alpha r_i / N}}
                 {\frac{1 - e^{-\alpha}}{e^{\alpha / N} - 1}}

    Minimal and maximal value depend on ``n``, ``N`` and ``alpha``:

    .. math::

        RIE_{min}(\alpha) = \frac{N}{n} \frac{1 - e^{\alpha n/N}}{1 - e^{\alpha}}

        RIE_{max}(\alpha) = \frac{N}{n} \frac{1 - e^{-\alpha n/N}}{1 - e^{-\alpha}}

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Target scores, e.g. probability of the positive class, or similarities to
        active compounds.

    alpha : float, default=20
        Exponential weight, roughly equivalent to ``1 / fraction`` from EF.

    References
    ----------
    .. [1] `Riniker, S., Landrum, G.A.
        "Open-source platform to benchmark fingerprints for ligand-based virtual screening"
        J Cheminform 5, 26 (2013).
        <https://doi.org/10.1186/1758-2946-5-26>`_

    .. [2] `Robert P. Sheridan et al.
        "Protocols for Bridging the Peptide to Nonpeptide Gap in Topological Similarity Searches"
        J. Chem. Inf. Comput. Sci. 2001, 41, 5, 1395-1406
        <https://doi.org/10.1021/ci0100144>`_

    .. [3] `Jean-François Truchon, Christopher I. Bayly
        "Evaluating Virtual Screening Methods: Good and Bad Metrics for the “Early Recognition” Problem"
        J. Chem. Inf. Model. 2007, 47, 2, 488-508
        <https://doi.org/10.1021/ci600426e>`_

    Returns
    -------
    score : float
        RIE value.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import rie_score
    >>> y_true = [0, 0, 1]
    >>> y_score = [0.1, 0.2, 0.7]
    >>> rie_score(y_true, y_score)  # doctest: +SKIP
    2.996182104771572
    """
    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError(f"RIE is only defined for binary y_true, got {y_type}")

    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    # sort by score only to keep original order for ties
    scores = sorted(zip(y_score, y_true, strict=True), key=itemgetter(0), reverse=True)
    return CalcRIE(scores, col=1, alpha=alpha)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "alpha": [Interval(Real, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def bedroc_score(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    alpha: float = 20,
) -> float:
    r"""
    Boltzmann-enhanced discrimination of ROC (BEDROC).

    A normalized alternative to Robust Initial Enhancement (RIE), scaled to have
    values in range ``[0, 1]``. ``alpha`` is directly passed to underlying RIE
    function, and is roughly equivalent to ``1 / fraction`` from enrichment factor
    (EF). See [1]_ [2]_ for details.

    Defined as:

    .. math::

        BEDROC(\alpha) = \frac{RIE(\alpha) - RIE_{min}(\alpha)}{RIE_{max}(\alpha) - RIE_{min}(\alpha)}

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Target scores, e.g. probability of the positive class, or similarities to
        active compounds.

    alpha : float, default=20
        Exponential weight, roughly equivalent to ``1 / fraction`` from EF.

    References
    ----------
    .. [1] `Riniker, S., Landrum, G.A.
        "Open-source platform to benchmark fingerprints for ligand-based virtual screening"
        J Cheminform 5, 26 (2013).
        <https://doi.org/10.1186/1758-2946-5-26>`_

    .. [2] `Jean-François Truchon, Christopher I. Bayly
        "Evaluating Virtual Screening Methods: Good and Bad Metrics for the “Early Recognition” Problem"
        J. Chem. Inf. Model. 2007, 47, 2, 488-508
        <https://doi.org/10.1021/ci600426e>`_

    Returns
    -------
    score : float
        BEDROC value.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import bedroc_score
    >>> y_true = [0, 0, 1]
    >>> y_score = [0.1, 0.2, 0.7]
    >>> bedroc_score(y_true, y_score)  # doctest: +SKIP
    0.9999999999999999
    """
    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError(f"BEDROC is only defined for binary y_true, got {y_type}")

    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    # sort by score only, so that ties keep the original input order; sorting the
    # (score, label) tuples would rank actives above inactives on equal scores,
    # optimistically inflating the metric
    scores = sorted(zip(y_score, y_true, strict=True), key=itemgetter(0), reverse=True)
    return CalcBEDROC(scores, col=1, alpha=alpha)
