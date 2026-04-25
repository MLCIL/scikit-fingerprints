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
def simpson_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Simpson similarity for vectors of binary values.

    Computes the Simpson similarity [1]_ (also known as asymmetric similarity [2]_ [3]_
    or overlap coefficient [4]_) for binary data between two input arrays or sparse
    matrices using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{\min(|a|, |b|)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    If any of the vectors is all-zeros, it results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Simpson similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
        "Mammals and the nature of continents."
        American Journal of Science, 241: 1-31 (1943).
        <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
        <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
    else:
        num_common = len(set(vec_a.indices) & set(vec_b.indices))  # type: ignore

    min_vec = min(np.sum(vec_a), np.sum(vec_b))

    sim = num_common / min_vec if min_vec != 0 else 0
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def simpson_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Simpson distance for vectors of binary values.

    Computes the Simpson distance for binary data between two input arrays
    or sparse matrices by subtracting the Simpson similarity [1]_ [2]_ [3]_ [4]_
    from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`simpson_binary_similarity`.
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
        Simpson distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
        "Mammals and the nature of continents."
        American Journal of Science, 241: 1-31 (1943).
        <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
        <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - simpson_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_simpson_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Simpson similarity for binary matrices.

    Computes the pairwise Simpson (also known as asymmetric similarity or overlap
    coefficient) similarity between binary matrices. If one array is passed,
    similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`simpson_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed,
        similarities are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Simpson similarity values. Shape is :math:`m \times n`
        if two arrays are passed, or :math:`m \times m` otherwise.

    Examples
    --------
    >>> from skfp.distances import bulk_simpson_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_simpson_binary_similarity(X, Y)
    >>> sim
    array([[1. , 0.5],
           [1. , 1. ]])

    >>> from scipy.sparse import csr_array
    >>> X = csr_array([[1, 0, 1], [0, 0, 1]])
    >>> Y = csr_array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_simpson_binary_similarity(X, Y)
    >>> sim
    array([[1. , 0.5],
           [1. , 1. ]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_simpson_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y)
        return _bulk_simpson_binary_similarity_two(X, Y)


def _bulk_simpson_binary_similarity_single(X: csr_array) -> np.ndarray:
    intersection = (X @ X.T).toarray()
    row_sums = np.array(X.sum(axis=1)).ravel()
    denom = np.minimum.outer(row_sums, row_sums)

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(intersection, denom, where=denom != 0)

    sims[denom == 0] = 0
    return sims


def _bulk_simpson_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.array(X.sum(axis=1)).ravel()
    row_sums_Y = np.array(Y.sum(axis=1)).ravel()
    denom = np.minimum(row_sums_X[:, None], row_sums_Y[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(intersection, denom, where=denom != 0)

    sims[denom == 0] = 0
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_simpson_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Simpson distance for binary matrices.

    Computes the pairwise Simpson distance between binary matrices. If one array
    is passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`simpson_binary_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed,
        distances are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Simpson distance values. Shape is :math:`m \times n`
        if two arrays are passed, or :math:`m \times m` otherwise.

    Examples
    --------
    >>> from skfp.distances import bulk_simpson_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_simpson_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> from scipy.sparse import csr_array
    >>> X = csr_array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_simpson_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_simpson_binary_similarity(X, Y)
