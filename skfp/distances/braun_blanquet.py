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
def braun_blanquet_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Braun-Blanquet similarity for vectors of binary values.

    Computes the Braun-Blanquet similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{\max(|a|, |b|)}

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
        Braun-Blanquet similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Braun-Blanquet, J.
        "Plant sociology. The study of plant communities. First ed."
        McGraw-Hill Book Co., Inc., New York and London, 1932.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19331600801>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import braun_blanquet_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = braun_blanquet_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[0, 0, 1]])
    >>> sim = braun_blanquet_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5
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

    max_vec = max(np.sum(vec_a), np.sum(vec_b))

    sim = float(num_common / max_vec) if max_vec != 0 else 1
    return sim


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def braun_blanquet_binary_distance(
    vec_a: list | np.ndarray | csr_array, vec_b: list | np.ndarray | csr_array
) -> float:
    """
    Braun-Blanquet distance for vectors of binary values.

    Computes the Braun-Blanquet distance for binary data between two input arrays
    or sparse matrices by subtracting the Braun-Blanquet similarity [1]_ [2]_ [3]_
    from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`braun_blanquet_binary_similarity`.
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
        Braun-Blanquet distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Braun-Blanquet, J.
        "Plant sociology. The study of plant communities. First ed."
        McGraw-Hill Book Co., Inc., New York and London, 1932.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19331600801>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import braun_blanquet_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = braun_blanquet_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = braun_blanquet_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - braun_blanquet_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_braun_blanquet_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Braun-Blanquet similarity for binary matrices.

    Computes the pairwise Braun-Blanquet similarity between binary matrices.
    If one array is passed, similarities are computed between its rows. For two arrays,
    similarities are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`braun_blanquet_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`. If not passed,
        similarities are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Braun-Blanquet similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`braun_blanquet_binary_similarity` : Braun-Blanquet similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_braun_blanquet_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_braun_blanquet_binary_similarity(X, Y)
    >>> sim
    array([[1. , 0.5],
           [0.5, 0.5]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_braun_blanquet_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y)
        return _bulk_braun_blanquet_binary_similarity_two(X, Y)


def _bulk_braun_blanquet_binary_similarity_single(X: csr_array) -> np.ndarray:
    intersection = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    max_denoms = np.maximum.outer(row_sums, row_sums)

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, max_denoms, out=sims, where=max_denoms != 0)
    np.fill_diagonal(sims, 1)

    return sims


def _bulk_braun_blanquet_binary_similarity_two(
    X: csr_array, Y: csr_array
) -> np.ndarray:
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()
    max_denoms = np.maximum.outer(row_sums_X, row_sums_Y)

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, max_denoms, out=sims, where=max_denoms != 0)

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_braun_blanquet_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Braun-Blanquet distance for vectors of binary values.

    Computes the pairwise Braun-Blanquet distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`braun_blanquet_binary_distance`.

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
        Array with pairwise Braun-Blanquet distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`braun_blanquet_binary_distance` : Braun-Blanquet distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_braun_blanquet_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_braun_blanquet_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_braun_blanquet_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_braun_blanquet_binary_similarity(X, Y)
