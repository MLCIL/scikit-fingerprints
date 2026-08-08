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
def kulczynski_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Kulczynski similarity for vectors of binary values.

    Computes the Kulczynski II similarity [1]_ [2]_ [3]_ for binary data between
    two input arrays or sparse matrices using the formula:

    .. math::

        sim(x, y) = \frac{1}{2} \left( \frac{a}{a+b} + \frac{a}{a+c} \right)

    where :math:`a`, :math:`b` and :math:`c` correspond to the number of bit
    relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1

    Note that this is the second Kulczynski similarity, also used by RDKit. It
    differs from Kulczynski I similarity from e.g. SciPy.

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing two all-zero vectors to this function results in a similarity of 1.
    However, when only one is all-zero (i.e. :math:`a+b=0` or :math:`a+c=0`), the
    similarity is 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Kulczynski similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1

    if isinstance(vec_a, list):
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)

    if isinstance(vec_a, np.ndarray):
        a = np.sum(np.logical_and(vec_a, vec_b))
        b = np.sum(np.logical_and(vec_a, 1 - vec_b))  # type: ignore
        c = np.sum(np.logical_and(1 - vec_a, vec_b))
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)

    if a + b == 0 or a + c == 0:
        return 0

    sim = (a / (a + b) + a / (a + c)) / 2
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def kulczynski_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Kulczynski distance for vectors of binary values.

    Computes the Kulczynski II distance for binary data between two input arrays
    or sparse matrices by subtracting the Kulczynski II similarity [1]_ [2]_ [3]_
    from 1, using to the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`kulczynski_binary_similarity`.
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
        Kulczynski distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - kulczynski_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_kulczynski_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Kulczynski similarity for binary matrices.

    Computes the pairwise Kulczynski similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`kulczynski_binary_similarity`.

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
        Array with pairwise Kulczynski similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`kulczynski_binary_similarity` : Kulczynski similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_kulczynski_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_kulczynski_binary_similarity(X, Y)
    >>> sim
    array([[1.  , 0.5 ],
           [0.75, 0.75]])
    """
    X = array_to_binary_csr(X)

    if Y is None:
        return _bulk_kulczynski_binary_similarity_single(X)
    else:
        Y = array_to_binary_csr(Y)
        return _bulk_kulczynski_binary_similarity_two(X, Y)


def _bulk_kulczynski_binary_similarity_single(X: csr_array) -> np.ndarray:
    # formula: 0.5 * (a/a+b + a/a+c)  # noqa: ERA001
    # note that:
    #   a = |A & B|, b = |A - B|, a+b = |A|
    #   c = |B - A|, a+c = |B|
    # we can rewrite formula row-wise as: 0.5 * (a / |A| + a / |B|)

    intersection = (X @ X.T).toarray()  # a = |A and B|
    row_sums = np.asarray(X.sum(axis=1)).ravel()

    denom_A = row_sums[:, None]  # |A|
    denom_B = row_sums[None, :]  # |B|

    term_A = np.zeros_like(intersection, dtype=float)
    term_B = np.zeros_like(intersection, dtype=float)

    np.divide(intersection, denom_A, out=term_A, where=denom_A != 0)
    np.divide(intersection, denom_B, out=term_B, where=denom_B != 0)
    sims = 0.5 * (term_A + term_B)

    both_zero = (denom_A == 0) & (denom_B == 0)
    sims[both_zero] = 1

    np.fill_diagonal(sims, 1)
    return sims


def _bulk_kulczynski_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()

    denom_A = row_sums_X[:, None]
    denom_B = row_sums_Y[None, :]

    term_A = np.zeros_like(intersection, dtype=float)
    term_B = np.zeros_like(intersection, dtype=float)

    np.divide(intersection, denom_A, out=term_A, where=denom_A != 0)
    np.divide(intersection, denom_B, out=term_B, where=denom_B != 0)
    sims = 0.5 * (term_A + term_B)

    both_zero = (denom_A == 0) & (denom_B == 0)
    sims[both_zero] = 1

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_kulczynski_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Kulczynski distance for vectors of binary values.

    Computes the pairwise Kulczynski distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`kulczynski_binary_distance`.

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
        Array with pairwise Kulczynski distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`kulczynski_binary_distance` : Kulczynski distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_kulczynski_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_kulczynski_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_kulczynski_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_kulczynski_binary_similarity(X, Y)
