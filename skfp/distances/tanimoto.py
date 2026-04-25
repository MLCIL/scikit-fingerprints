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
def tanimoto_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Tanimoto similarity for vectors of binary values.

    Computes the Tanimoto similarity [1]_ (also known as Jaccard similarity)
    for binary data between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{|a \cup b|} = \frac{|a \cap b|}{|a| + |b| - |a \cap b|}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : ndarray or CSR sparse matrix
        First binary input array or sparse matrix.

    vec_b : ndarray or CSR sparse matrix
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Tanimoto similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    --------
    >>> from skfp.distances import tanimoto_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        union = np.sum(np.logical_or(vec_a, vec_b))
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore
        intersection = len(vec_a_idxs & vec_b_idxs)
        union = len(vec_a_idxs | vec_b_idxs)

    sim = intersection / union if union != 0 else 1
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Tanimoto distance for vectors of binary values.

    Computes the Tanimoto distance [1]_ (also known as Jaccard distance)
    for binary data between two input arrays or sparse matrices by subtracting
    the similarity from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`tanimoto_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : ndarray or CSR sparse matrix
        First binary input array or sparse matrix.

    vec_b : ndarray or CSR sparse matrix
        Second binary input array or sparse matrix.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Returns
    -------
    distance : float
        Tanimoto distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import tanimoto_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - tanimoto_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_count_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Tanimoto similarity for vectors of count values.

    Computes the Tanimoto similarity [1]_ for count data between two input arrays
    or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{a \cdot b}{\|a\|^2 + \|b\|^2 - a \cdot b}

    Calculated similarity falls within the range of :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Parameters
    ----------
    vec_a : ndarray or CSR sparse matrix
        First count input array or sparse matrix.

    vec_b : ndarray or CSR sparse matrix
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        Tanimoto similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    --------
    >>> from skfp.distances import tanimoto_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    0.9811320754716981

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    0.9811320754716981
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

    intersection = dot_ab
    union = dot_aa + dot_bb - dot_ab

    sim = intersection / union if union >= 1e-8 else 1
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_count_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Tanimoto distance for vectors of count values.

    Computes the Tanimoto distance [1]_ for binary data between two input arrays
    or sparse matrices by subtracting similarity value from 1, using the formula:

    .. math::

            dist(a, b) = 1 - sim(a, b)

    See also :py:func:`tanimoto_count_similarity`.
    Calculated distance falls within the range from :math:`[0, 1]`.
    Passing all-zero vectors to this function results in distance of 0.

    Parameters
    ----------
    vec_a : ndarray or CSR sparse matrix
        First count input array or sparse matrix.

    vec_b : ndarray or CSR sparse matrix
        Second count input array or sparse matrix.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Returns
    -------
    distance : float
        Tanimoto distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import tanimoto_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.018867924528301883

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.018867924528301883
    """
    return 1 - tanimoto_count_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto similarity for binary matrices.

    Computes the pairwise Tanimoto similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`tanimoto_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, similarities are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Tanimoto similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_binary_similarity` : Tanimoto similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_tanimoto_binary_similarity(X, Y)
    >>> sim
    array([[1.        , 0.33333333],
           [0.5       , 0.5       ]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_tanimoto_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)

        return _bulk_tanimoto_binary_similarity_two(X, Y)


def _bulk_tanimoto_binary_similarity_single(X: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # union = |x| + |y| - intersection, |x| is number of 1s
    intersection = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    unions = np.add.outer(row_sums, row_sums) - intersection

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, unions, out=sims, where=unions != 0)

    return sims


def _bulk_tanimoto_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # union = |x| + |y| - intersection, |x| is number of 1s
    intersection = (X @ Y.T).toarray()

    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()

    unions = np.add.outer(row_sums_X, row_sums_Y) - intersection

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, unions, out=sims, where=unions != 0)

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto distance for vectors of binary values.

    Computes the pairwise Tanimoto distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`tanimoto_binary_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, distances are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Tanimoto distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_binary_distance` : Tanimoto distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_tanimoto_binary_similarity(X, Y)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_count_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto similarity for count matrices.

    Computes the pairwise Tanimoto similarity between count matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`tanimoto_count_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, similarities are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Tanimoto similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_count_similarity` : Tanimoto similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_count_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_tanimoto_count_similarity(X, Y)
    >>> sim
    array([[1.        , 0.33333333],
           [0.5       , 0.5       ]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_tanimoto_count_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)

        return _bulk_tanimoto_count_similarity_two(X, Y)


def _bulk_tanimoto_count_similarity_single(X: csr_array) -> np.ndarray:
    # union for counts = dot(x,x) + dot(y,y) - dot(x,y)
    intersection = (X @ X.T).toarray()
    row_norms = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    unions = np.add.outer(row_norms, row_norms) - intersection

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, unions, out=sims, where=unions >= 1e-8)

    np.fill_diagonal(sims, 1)

    return sims


def _bulk_tanimoto_count_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # union for counts = dot(x,x) + dot(y,y) - dot(x,y)
    intersection = (X @ Y.T).toarray()
    row_norms_X = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    row_norms_Y = np.asarray(Y.multiply(Y).sum(axis=1)).ravel()

    unions = np.add.outer(row_norms_X, row_norms_Y) - intersection

    sims = np.empty_like(intersection, dtype=float)
    np.divide(intersection, unions, out=sims, where=unions >= 1e-8)

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_count_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto distance for vectors of count values.

    Computes the pairwise Tanimoto distance between count matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`tanimoto_count_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, distances are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Tanimoto distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_count_distance` : Tanimoto distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_count_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_count_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_count_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_tanimoto_count_similarity(X, Y)
