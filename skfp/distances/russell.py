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
def russell_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Russell similarity for vectors of binary values.

    Computes the Russell similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a}{n}

    where

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`n` - length of passed vectors

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, CSR sparse array}
        First binary input array or sparse matrix.

    vec_b : {ndarray, CSR sparse array}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Russell similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Russell P.F., Rao T.R.
        "On habitat and association of species of anopheline larvae in south-eastern Madras"
        Journal of the Malaria Institute of India, 1940, June, Vol. 3, No. 1, 153-178 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19412900343>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import russell_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        a = np.sum(np.logical_and(vec_a, vec_b))
        n = len(vec_a)
    else:
        n = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore
        a = len(vec_a_idxs & vec_b_idxs)

    return float(a / n)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def russell_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Russell distance for vectors of binary values.

    Computes the Russell distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`russell_binary_similarity`.
    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, CSR sparse array}
        First binary input array or sparse matrix.

    vec_b : {ndarray, CSR sparse array}
        Second binary input array or sparse matrix.

    Returns
    -------
    distance : float
        Russell distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Russell P.F., Rao T.R.
        "On habitat and association of species of anopheline larvae in south-eastern Madras"
        Journal of the Malaria Institute of India, 1940, June, Vol. 3, No. 1, 153-178 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19412900343>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import russell_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.5

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.5
    """
    return 1 - russell_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_russell_binary_similarity(
    X: list | np.ndarray | csr_array,
    Y: list | np.ndarray | csr_array | None = None,
) -> np.ndarray:
    r"""
    Bulk Russell similarity for binary matrices.

    Computes the pairwise Russell similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`russell_binary_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array, of shape :math:`m \times d`

    Y : ndarray or CSR sparse array, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed, similarities
        are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Russell similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_russell_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y)
        return _bulk_russell_binary_similarity_two(X, Y)


def _bulk_russell_binary_similarity_single(X: csr_array) -> np.ndarray:
    n_features = X.shape[1]
    a = (X @ X.T).toarray()
    sims = a / n_features
    return sims


def _bulk_russell_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    n_features = X.shape[1]
    a = (X @ Y.T).toarray()
    sims = a / n_features
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_russell_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Russell distance for vectors of binary values.

    Computes the pairwise Russell distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`russell_binary_distance`.

    Parameters
    ----------
    X : ndarray
        First binary input array, of shape :math:`m \times d`

    Y : ndarray, default=None
        Second binary input array, of shape :math:`n \times d`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Russell distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`russell_binary_distance` : Russell distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_russell_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_russell_binary_distance(X, Y)
    >>> dist
    array([[0.33333333, 0.33333333],
           [0.33333333, 0.66666667]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_russell_binary_distance(X)
    >>> dist
    array([[0.        , 0.66666667],
           [0.66666667, 0.66666667]])
    """
    return 1 - bulk_russell_binary_similarity(X, Y)
