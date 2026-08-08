import numpy as np
from scipy.sparse import csr_array
from sklearn.utils._param_validation import validate_params

from skfp.distances.utils import array_to_binary_csr


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def rogot_goldberg_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Rogot-Goldberg similarity for vectors of binary values.

    Computes the Rogot-Goldberg similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a}{2 * (2a + b + c)} +
                    \frac{d}{2 * (2d + b + c)}

    where :math:`a`, :math:`b`, :math:`c` and :math:`d` correspond to the number
    of bit relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1
    - :math:`d` - both are 0

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Rogot-Goldberg similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rogot E., Goldberg I.D.
        "A proposed index for measuring agreement in test-retest studies."
        Journal of Chronic Diseases 19.9 (1966): 991-1006.`
        <https://doi.org/10.1016/0021-9681(66)90032-4>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rogot_goldberg_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = rogot_goldberg_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = rogot_goldberg_binary_similarity(vec_a, vec_b)
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
        vec_a_neg = 1 - vec_a
        vec_b_neg = 1 - vec_b  # type: ignore

        a = np.sum(np.logical_and(vec_a, vec_b))
        b = np.sum(np.logical_and(vec_a, vec_b_neg))
        c = np.sum(np.logical_and(vec_a_neg, vec_b))
        d = np.sum(np.logical_and(vec_a_neg, vec_b_neg))
    else:
        length = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)
        d = length - (a + b + c)

    first_denom = 2 * a + b + c
    second_denom = 2 * d + b + c

    # all-ones or all-zeros vectors
    if first_denom == 0 or second_denom == 0:
        return 1

    sim = a / first_denom + d / second_denom

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def rogot_goldberg_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Rogot-Goldberg distance for vectors of binary values.

    Computes the Rogot-Goldberg distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`rogot_goldberg_binary_similarity`.
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
    distance : float
        Rogot-Goldberg distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rogot E., Goldberg I.D.
        "A proposed index for measuring agreement in test-retest studies."
        Journal of Chronic Diseases 19.9 (1966): 991-1006.`
        <https://doi.org/10.1016/0021-9681(66)90032-4>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rogot_goldberg_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = rogot_goldberg_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = rogot_goldberg_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - rogot_goldberg_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_rogot_goldberg_binary_similarity(
    X: list | np.ndarray | csr_array,
    Y: list | np.ndarray | csr_array | None = None,
) -> np.ndarray:
    r"""
    Bulk Rogot-Goldberg similarity for binary matrices.

    Computes the pairwise Rogot-Goldberg similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`rogot_goldberg_binary_similarity`.

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
        Array with pairwise Rogot-Goldberg similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`rogot_goldberg_binary_similarity` : Rogot-Goldberg similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_rogot_goldberg_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_rogot_goldberg_binary_similarity(X, Y)
    >>> sim
    array([[0.4       , 0.4       ],
           [0.66666667, 0.66666667]])
    """
    X = array_to_binary_csr(X)

    if Y is None:
        return _bulk_rogot_goldberg_binary_similarity_single(X)
    else:
        Y = array_to_binary_csr(Y)
        return _bulk_rogot_goldberg_binary_similarity_two(X, Y)


def _bulk_rogot_goldberg_binary_similarity_single(X: csr_array) -> np.ndarray:
    n_features = X.shape[1]

    # a - intersection
    a = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()

    # b+c = |A| + |B| - 2a
    sum_A = row_sums[:, None]
    sum_B = row_sums[None, :]
    bc_sum = sum_A + sum_B - 2 * a

    # d = n - (a + b + c)  # noqa: ERA001
    d = n_features - (a + bc_sum)

    denom_1 = 2 * a + bc_sum
    denom_2 = 2 * d + bc_sum

    sims = np.zeros_like(a, dtype=float)
    np.divide(a, denom_1, out=sims, where=denom_1 != 0)

    part_2 = np.zeros_like(a, dtype=float)
    np.divide(d, denom_2, out=part_2, where=denom_2 != 0)
    sims += part_2

    denom_zero = (denom_1 == 0) | (denom_2 == 0)
    sims[denom_zero] = 1

    return sims


def _bulk_rogot_goldberg_binary_similarity_two(
    X: csr_array, Y: csr_array
) -> np.ndarray:
    n_features = X.shape[1]

    a = (X @ Y.T).toarray()

    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()

    sum_A = row_sums_X[:, None]
    sum_B = row_sums_Y[None, :]
    bc = sum_A + sum_B - 2 * a
    d = n_features - (a + bc)

    denom_1 = 2 * a + bc
    denom_2 = 2 * d + bc

    sims = np.zeros_like(a, dtype=float)
    np.divide(a, denom_1, out=sims, where=denom_1 != 0)

    part_2 = np.zeros_like(a, dtype=float)
    np.divide(d, denom_2, out=part_2, where=denom_2 != 0)

    sims += part_2

    denom_zero = (denom_1 == 0) | (denom_2 == 0)
    sims[denom_zero] = 1

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_rogot_goldberg_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Rogot-Goldberg distance for vectors of binary values.

    Computes the pairwise Rogot-Goldberg distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`rogot_goldberg_binary_distance`.

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
        Array with pairwise Rogot-Goldberg distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`rogot_goldberg_binary_distance` : Rogot-Goldberg distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_rogot_goldberg_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_rogot_goldberg_binary_distance(X, Y)
    >>> dist
    array([[0.6 , 0.6 ],
           [0.  , 0.75]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_rogot_goldberg_binary_distance(X)
    >>> dist
    array([[0.  , 0.75],
           [0.75, 0.  ]])
    """
    return 1 - bulk_rogot_goldberg_binary_similarity(X, Y)
