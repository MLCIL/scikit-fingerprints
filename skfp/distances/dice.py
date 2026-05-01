import numpy as np
from scipy.sparse import csr_array
from sklearn.utils._param_validation import validate_params


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def dice_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Dice similarity for vectors of binary values.

    Computes the Dice similarity [1]_ [2]_ [3]_ for binary data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(a, b) = \frac{2 |a \cap b|}{|a| + |b|}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Dice similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>`_

    .. [2] `Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>`_

    .. [3] `Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import dice_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = dice_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = dice_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, list):
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)

    if isinstance(vec_a, np.ndarray):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
    else:
        intersection = len(set(vec_a.indices) & set(vec_b.indices))  # type: ignore

    denominator = vec_a.sum() + vec_b.sum()  # type: ignore
    sim = 2 * intersection / denominator if denominator != 0 else 1

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def dice_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Dice distance for vectors of binary values.

    Computes the Dice distance for binary data between two input arrays
    or sparse matrices by subtracting the Dice similarity [1]_ [2]_ [3]_ from 1,
    using to the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`dice_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    distance : float
        Dice distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>`_

    .. [2] `Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>`_

    .. [3] `Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import dice_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = dice_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = dice_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - dice_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def dice_count_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Dice similarity for vectors of count values.

    Computes the Dice similarity [1]_ [2]_ [3]_ for count data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(a, b) = \frac{2 \cdot a \cdot b}{|a|^2 + |b|^2}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        Dice similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>`_

    .. [2] `Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>`_

    .. [3] `Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import dice_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = dice_count_similarity(vec_a, vec_b)
    >>> sim
    0.9904761904761905

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = dice_count_similarity(vec_a, vec_b)
    >>> sim
    0.9904761904761905
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        dot_aa = np.dot(vec_a, vec_a)
        dot_bb = np.dot(vec_b, vec_b)
        dot_ab = np.dot(vec_a, vec_b)
    else:
        dot_ab = vec_a.multiply(vec_b).sum()
        dot_aa = vec_a.multiply(vec_a).sum()
        dot_bb = vec_b.multiply(vec_b).sum()  # type: ignore

    denominator = dot_aa + dot_bb
    sim = 2 * dot_ab / denominator if denominator >= 1e-8 else 1

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def dice_count_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Dice distance for vectors of count values.

    Computes the Dice distance for count data between two input arrays
    or sparse matrices by subtracting the Dice similarity [1]_ [2]_ [3]_ from 1,
    using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`dice_count_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    distance : float
        Dice distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>`_

    .. [2] `Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>`_

    .. [3] `Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import dice_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = dice_count_distance(vec_a, vec_b)
    >>> dist
    0.00952380952380949

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = dice_count_distance(vec_a, vec_b)
    >>> dist
    0.00952380952380949
    """
    return 1 - dice_count_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_dice_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Dice similarity for binary matrices.

    Computes the pairwise Dice similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`dice_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`. If not passed, similarities
        are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Dice similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`dice_binary_similarity` : Dice similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_dice_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_dice_binary_similarity(X, Y)
    >>> sim
    array([[1.        , 0.5       ],
           [0.66666667, 0.66666667]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_dice_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)
        return _bulk_dice_binary_similarity_two(X, Y)


def _bulk_dice_binary_similarity_single(X: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # |x| + |y| = sum of 1s in both rows
    intersection = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    denom = np.add.outer(row_sums, row_sums)

    sims = np.empty_like(intersection, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.multiply(2, intersection, out=sims)
        np.divide(sims, denom, out=sims, where=denom != 0)

    sims[denom == 0] = 1
    np.fill_diagonal(sims, 1)

    return sims


def _bulk_dice_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # |x| + |y| = sum of 1s in both rows
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()
    denom = np.add.outer(row_sums_X, row_sums_Y)

    sims = np.empty_like(intersection, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.multiply(2, intersection, out=sims)
        np.divide(sims, denom, out=sims, where=denom != 0)

    sims[denom == 0] = 1
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_dice_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Dice distance for vectors of binary values.

    Computes the pairwise Dice distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`dice_binary_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Dice distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`dice_binary_distance` : Dice distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_dice_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_dice_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_dice_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_dice_binary_similarity(X, Y)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_dice_count_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Dice similarity for count matrices.

    Computes the pairwise Dice similarity between count matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`dice_count_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First count input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second count input array or sparse matrix, of shape :math:`n \times d`. If not passed, similarities
        are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Dice similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`dice_count_similarity` : Dice similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_dice_count_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_dice_count_similarity(X, Y)
    >>> sim
    array([[1.        , 0.5       ],
           [0.66666667, 0.66666667]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_dice_count_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)
        return _bulk_dice_count_similarity_two(X, Y)


def _bulk_dice_count_similarity_single(X: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # |x| + |y| = dot(x,x) + dot(y,y)
    dot_products = (X @ X.T).toarray()
    dot_self = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    denom = np.add.outer(dot_self, dot_self)

    sims = np.empty_like(dot_products, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.multiply(2, dot_products, out=sims)
        np.divide(sims, denom, out=sims, where=denom >= 1e-8)

    sims[denom < 1e-8] = 1
    np.fill_diagonal(sims, 1)
    return sims


def _bulk_dice_count_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # |x| + |y| = dot(x,x) + dot(y,y)
    dot_products = (X @ Y.T).toarray()
    dot_self_X = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    dot_self_Y = np.asarray(Y.multiply(Y).sum(axis=1)).ravel()
    denom = np.add.outer(dot_self_X, dot_self_Y)

    sims = np.empty_like(dot_products, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.multiply(2, dot_products, out=sims)
        np.divide(sims, denom, out=sims, where=denom >= 1e-8)

    sims[denom < 1e-8] = 1
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_dice_count_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Dice distance for vectors of count values.

    Computes the pairwise Dice distance between count matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`dice_count_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First count input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second count input array or sparse matrix, of shape :math:`n \times d`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Dice distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`dice_count_distance` : Dice distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_dice_count_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_dice_count_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_dice_count_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_dice_count_similarity(X, Y)
