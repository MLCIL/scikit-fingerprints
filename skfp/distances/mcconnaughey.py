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
def mcconnaughey_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
    normalized: bool = False,
) -> float:
    r"""
    McConnaughey similarity for vectors of binary values.

    Computes the McConnaughey similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{(|a \cap b| \cdot (|a| + |b|) - |a| \cdot |b|}{|a| \cdot |b|}
                  = \frac{|a \cap b|}{|a|} + \frac{|a \cap b|}{|b|} - 1

    The calculated similarity falls within the range :math:`[-1, 1]`.
    Use ``normalized`` argument to scale the similarity to range :math:`[0, 1]`.
    Passing two all-zero vectors to this function results in a similarity of 1. Passing
    only one all-zero vector results in a similarity of -1 for the non-normalized variant
    and 0 for the normalized variant.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    normalized : bool, default=False
        Whether to normalize values to range ``[0, 1]`` by adding one and dividing the result
        by 2.

    Returns
    -------
    similarity : float
        McConnaughey similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `McConnaughey B.H.
        "The determination and analysis of plankton communities"
        Lembaga Penelitian Laut, 1964.
        <https://books.google.pl/books?id=7aBbOQAACAAJ>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import mcconnaughey_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = mcconnaughey_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = mcconnaughey_binary_similarity(vec_a, vec_b)
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
        vec_a_ones = np.sum(vec_a)
        vec_b_ones = np.sum(vec_b)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore

        num_common = len(vec_a_idxs & vec_b_idxs)
        vec_a_ones = len(vec_a_idxs)
        vec_b_ones = len(vec_b_idxs)

    sum_ab_ones = vec_a_ones + vec_b_ones
    dot_ab_ones = vec_a_ones * vec_b_ones

    if sum_ab_ones == 0:
        sim = 1.0
    elif dot_ab_ones == 0:
        sim = -1.0
    else:
        sim = (num_common * sum_ab_ones - dot_ab_ones) / dot_ab_ones

    if normalized:
        sim = (sim + 1) / 2

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def mcconnaughey_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    McConnaughey distance for vectors of binary values.

    Computes the McConnaughey distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`mcconnaughey_binary_similarity`. It uses the normalized
    similarity, scaled to range ``[0, 1]``.
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
        McConnaughey distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `McConnaughey B.H.
        "The determination and analysis of plankton communities"
        Lembaga Penelitian Laut, 1964.
        <https://books.google.pl/books?id=7aBbOQAACAAJ>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import mcconnaughey_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = mcconnaughey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = mcconnaughey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - mcconnaughey_binary_similarity(vec_a, vec_b, normalized=True)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_mcconnaughey_binary_similarity(
    X: list | np.ndarray | csr_array,
    Y: list | np.ndarray | csr_array | None = None,
    normalized: bool = False,
) -> np.ndarray:
    r"""
    Bulk McConnaughey similarity for binary matrices.

    Computes the pairwise McConnaughey similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`mcconnaughey_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed, similarities
        are computed between rows of X.

    normalized : bool, default=False
        Whether to normalize the values inside the result matrix to range ``[0, 1]`` by adding
        one and dividing the result by 2.

    Returns
    -------
    similarities : ndarray
        Array with pairwise McConnaughey similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcconnaughey_binary_similarity` : McConnaughey similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_mcconnaughey_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_mcconnaughey_binary_similarity(X, Y)
    >>> sim
    array([[0.66666667, 0.66666667],
           [0.5       , 0.5       ]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_mcconnaughey_binary_similarity_single(X, normalized)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y)
        return _bulk_mcconnaughey_binary_similarity_two(X, Y, normalized)


def _bulk_mcconnaughey_binary_similarity_single(
    X: csr_array, normalized: bool
) -> np.ndarray:
    intersection = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()

    denom_A = row_sums[:, None]
    denom_B = row_sums[None, :]

    term_A = np.zeros_like(intersection, dtype=float)
    term_B = np.zeros_like(intersection, dtype=float)

    np.divide(intersection, denom_A, out=term_A, where=denom_A != 0)
    np.divide(intersection, denom_B, out=term_B, where=denom_B != 0)

    sims = term_A + term_B - 1

    both_zero = (denom_A == 0) & (denom_B == 0)
    sims[both_zero] = 1

    if normalized:
        sims = (sims + 1) / 2

    np.fill_diagonal(sims, 1)
    return sims


def _bulk_mcconnaughey_binary_similarity_two(
    X: csr_array, Y: csr_array, normalized: bool
) -> np.ndarray:
    intersection = (X @ Y.T).toarray()

    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()

    denom_A = row_sums_X[:, None]
    denom_B = row_sums_Y[None, :]

    term_A = np.zeros_like(intersection, dtype=float)
    term_B = np.zeros_like(intersection, dtype=float)

    np.divide(intersection, denom_A, out=term_A, where=denom_A != 0)
    np.divide(intersection, denom_B, out=term_B, where=denom_B != 0)

    sims = term_A + term_B - 1

    both_zero = (denom_A == 0) & (denom_B == 0)
    sims[both_zero] = 1

    if normalized:
        sims = (sims + 1) / 2

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_mcconnaughey_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk McConnaughey distance for vectors of binary values.

    Computes the pairwise McConnaughey distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`mcconnaughey_binary_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise McConnaughey distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcconnaughey_binary_distance` : McConnaughey distance function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_mcconnaughey_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_mcconnaughey_binary_distance(X, Y)
    >>> dist
    array([[0.16666667, 0.16666667],
           [0.        , 0.5       ]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_mcconnaughey_binary_distance(X)
    >>> dist
    array([[0.        , 0.33333333],
           [0.33333333, 0.        ]])
    """
    return 1 - bulk_mcconnaughey_binary_similarity(X, Y, normalized=True)
