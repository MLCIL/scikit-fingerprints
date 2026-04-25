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
def sokal_sneath_2_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Sokal-Sneath similarity 2 for vectors of binary values.

    Computes the Sokal-Sneath similarity 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{|a \cup b| + |a \Delta b|} =
                    \frac{|a \cap b|}{2 * |a| + 2 * |b| - 3 * |a \cap b|}

    where :`|a \Delta b|` is the XOR operation (symmetric difference), i.e. number
    of bits that are "on" in one vector and "off" in another.

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Sokal-Sneath similarity 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        a_sum = np.sum(vec_a)
        b_sum = np.sum(vec_b)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)  # type: ignore

        intersection = len(vec_a_idxs & vec_b_idxs)
        a_sum = len(vec_a_idxs)
        b_sum = len(vec_b_idxs)

    denominator = 2 * a_sum + 2 * b_sum - 3 * intersection
    sim = intersection / denominator if denominator > 0 else 1.0

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def sokal_sneath_2_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Sokal-Sneath distance 2 for vectors of binary values.

    Computes the Sokal-Sneath distance 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`sokal_sneath_2_binary_similarity`.
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
        Sokal-Sneath distance 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667
    """
    return 1 - sokal_sneath_2_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_sokal_sneath_2_binary_similarity(
    X: list | np.ndarray | csr_array,
    Y: list | np.ndarray | csr_array | None = None,
) -> np.ndarray:
    r"""
    Bulk Sokal-Sneath similarity 2 for binary matrices.

    Computes the pairwise Sokal-Sneath similarity 2 between binary matrices.
    If one array is passed, similarities are computed between its rows.
    For two arrays, similarities are between their respective rows, with
    `i`-th row and `j`-th column in output corresponding to `i`-th row from the
    first array and `j`-th row from the second array.

    The formula is:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{2|a| + 2|b| - 3|a \cap b|}

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
        Array with pairwise Sokal-Sneath similarity 2 values. Shape is
        :math:`m \times n` if two arrays are passed, or :math:`m \times m`
        otherwise.

    Examples
    --------
    >>> from skfp.distances import bulk_sokal_sneath_2_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> bulk_sokal_sneath_2_binary_similarity(X, Y)
    array([[0.5       , 0.5       ],
           [0.33333333, 0.33333333]])

    >>> from scipy.sparse import csr_array
    >>> X = csr_array([[1, 1, 1], [0, 0, 1]])
    >>> Y = csr_array([[1, 0, 1], [0, 1, 1]])
    >>> bulk_sokal_sneath_2_binary_similarity(X, Y)
    array([[0.5       , 0.5       ],
           [0.33333333, 0.33333333]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_sokal_sneath_2_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y)
        return _bulk_sokal_sneath_2_binary_similarity_two(X, Y)


def _bulk_sokal_sneath_2_binary_similarity_single(X: csr_array) -> np.ndarray:
    intersection = (X @ X.T).toarray()
    row_sums = np.array(X.sum(axis=1)).ravel()

    sum_A = row_sums[:, None]
    sum_B = row_sums[None, :]
    denominator = 2 * sum_A + 2 * sum_B - 3 * intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(intersection, denominator, where=denominator > 0)

    sims[denominator == 0] = 1
    np.fill_diagonal(sims, 1)

    return sims


def _bulk_sokal_sneath_2_binary_similarity_two(
    X: csr_array, Y: csr_array
) -> np.ndarray:
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.array(X.sum(axis=1)).ravel()
    row_sums_Y = np.array(Y.sum(axis=1)).ravel()

    sum_A = row_sums_X[:, None]
    sum_B = row_sums_Y[None, :]
    denominator = 2 * sum_A + 2 * sum_B - 3 * intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(intersection, denominator, where=denominator > 0)

    sims[denominator == 0] = 1

    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_sokal_sneath_2_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Sokal-Sneath distance 2 for binary matrices.

    Computes the pairwise Sokal-Sneath distance 2 between binary matrices. If
    one array is passed, distances are computed between its rows. For two arrays,
    distances are between their respective rows, with `i`-th row and `j`-th
    column in output corresponding to `i`-th row from the first array and `j`-th row
    from the second array.

    See also :py:func:`sokal_sneath_2_binary_distance`.

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
        Array with pairwise Sokal-Sneath distance 2 values. Shape is
        :math:`m \times n` if two arrays are passed, or :math:`m \times m`
        otherwise.https://github.com/scikit-fingerprints/scikit-fingerprints/pull/488

    Examples
    --------
    >>> from skfp.distances import bulk_sokal_sneath_2_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> bulk_sokal_sneath_2_binary_distance(X, Y)
    array([[0.5, 0.5],
           [0. , 0.8]])

    >>> from scipy.sparse import csr_array
    >>> X = csr_array([[1, 1, 1], [1, 0, 1]])
    >>> Y = csr_array([[1, 0, 1], [1, 1, 0]])
    >>> bulk_sokal_sneath_2_binary_distance(X, Y)
    array([[0.5, 0.5],
           [0. , 0.8]])
    """
    return 1 - bulk_sokal_sneath_2_binary_similarity(X, Y)
