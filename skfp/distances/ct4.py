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
def ct4_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of binary values.

    Computes the Consonni–Todeschini 4 (CT4) similarity [1]_ [2]_ [3]_ for binary data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log (1 + |a \cap b|)}{\log (1 + |a \cup b|)}
        = \frac{\log (1 + |a \cap b|)}{\log (1 + |a| + |b| - |a \cap b|)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        CT4 similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import ct4_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = ct4_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = ct4_binary_similarity(vec_a, vec_b)
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

    sim = float(np.log1p(intersection) / np.log1p(union)) if union != 0 else 1
    return sim


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Consonni–Todeschini 4 distance for vectors of binary values.

    Computes the Consonni–Todeschini 4 (CT4) distance [1]_ [2]_ [3]_ for binary data
    between two input arrays or sparse matrices by subtracting the similarity
    from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`ct4_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Returns
    -------
    distance : float
        CT4 distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import ct4_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = ct4_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = ct4_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - ct4_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def ct4_count_similarity(
    vec_a: list | np.ndarray | csr_array, vec_b: list | np.ndarray | csr_array
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of count values.

    Computes the Consonni–Todeschini 4 (CT4) similarity [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log (1 + a \cdot b)}{\log (1 + \|a\|^2 + \|b\|^2 - a \cdot b)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        CT4 similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import ct4_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = ct4_count_similarity(vec_a, vec_b)
    >>> sim
    0.9953140617275088

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = ct4_count_similarity(vec_a, vec_b)
    >>> sim
    0.9953140617275088
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

    numerator = np.log1p(dot_ab)
    denominator = np.log1p(dot_aa + dot_bb - dot_ab)

    sim = float(numerator / denominator) if denominator >= 1e-8 else 1

    return sim


@validate_params(
    {
        "vec_a": ["array-like", "sparse matrix"],
        "vec_b": ["array-like", "sparse matrix"],
    },
    prefer_skip_nested_validation=True,
)
def ct4_count_distance(
    vec_a: list | np.ndarray | csr_array, vec_b: list | np.ndarray | csr_array
) -> float:
    """
    Consonni–Todeschini distance for vectors of count values.

    Computes the Consonni–Todeschini 4 (CT4) distance [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices by subtracting the similarity
    from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`ct4_count_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Vectors with 0 or 1 elements in their intersection or union (which
    would cause numerical problems with logarithm) have distance 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Returns
    -------
    distance : float
        CT4 distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import ct4_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = ct4_count_distance(vec_a, vec_b)
    >>> dist
    0.004685938272491197

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = ct4_count_distance(vec_a, vec_b)
    >>> dist
    0.004685938272491197
    """
    return 1 - ct4_count_similarity(vec_a, vec_b)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_binary_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 similarity for binary matrices.

    Computes the pairwise Consonni–Todeschini 4 (CT4) similarity between binary matrices.
    If one array is passed, similarities are computed between its rows. For two arrays,
    similarities are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`ct4_binary_similarity`.

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
        Array with pairwise CT4 similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    Examples
    --------
    >>> from skfp.distances import bulk_ct4_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_ct4_binary_similarity(X, Y)
    >>> sim
    array([[1.        , 0.5       ],
           [0.63092975, 0.63092975]])
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_ct4_binary_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)
        return _bulk_ct4_binary_similarity_two(X, Y)


def _bulk_ct4_binary_similarity_single(X: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # union = |x| + |y| - intersection, |x| is number of 1s
    intersection = (X @ X.T).toarray()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    unions = np.add.outer(row_sums, row_sums) - intersection

    sims = np.empty_like(intersection, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(np.log1p(intersection), np.log1p(unions), out=sims, where=unions != 0)

    sims[unions == 0] = 1
    np.fill_diagonal(sims, 1)
    return sims


def _bulk_ct4_binary_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # intersection = x * y, dot product
    # union = |x| + |y| - intersection, |x| is number of 1s
    intersection = (X @ Y.T).toarray()
    row_sums_X = np.asarray(X.sum(axis=1)).ravel()
    row_sums_Y = np.asarray(Y.sum(axis=1)).ravel()
    unions = np.add.outer(row_sums_X, row_sums_Y) - intersection

    sims = np.empty_like(intersection, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(np.log1p(intersection), np.log1p(unions), out=sims, where=unions != 0)

    sims[unions == 0] = 1
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_binary_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 distance for vectors of binary values.

    Computes the pairwise Consonni–Todeschini 4 (CT4) distance between binary matrices.
    If one array is passed, distances are computed between its rows. For two arrays,
    distances are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`ct4_binary_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First binary input array or sparse matrix, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second binary input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, distances are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise CT4 distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.
    """
    return 1 - bulk_ct4_binary_similarity(X, Y)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_count_similarity(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 similarity for count matrices.

    Computes the pairwise Consonni–Todeschini 4 similarity between count matrices.
    If one array is passed, similarities are computed between its rows. For two arrays,
    similarities are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`ct4_count_similarity`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First count input array or sparse matrix, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second count input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, similarities are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise CT4 similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    if Y is None:
        return _bulk_ct4_count_similarity_single(X)
    else:
        if not isinstance(Y, csr_array):
            Y = csr_array(Y, dtype=float)
        return _bulk_ct4_count_similarity_two(X, Y)


def _bulk_ct4_count_similarity_single(X: csr_array) -> np.ndarray:
    # union for counts = dot(x,x) + dot(y,y) - dot(x,y)
    dot_products = (X @ X.T).toarray()
    dot_self = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    unions = np.add.outer(dot_self, dot_self) - dot_products

    sims = np.empty_like(dot_products, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            np.log1p(dot_products), np.log1p(unions), out=sims, where=unions >= 1e-8
        )

    sims[unions == 0] = 1
    np.fill_diagonal(sims, 1)
    return sims


def _bulk_ct4_count_similarity_two(X: csr_array, Y: csr_array) -> np.ndarray:
    # union for counts = dot(x,x) + dot(y,y) - dot(x,y)
    dot_products = (X @ Y.T).toarray()
    dot_self_X = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    dot_self_Y = np.asarray(Y.multiply(Y).sum(axis=1)).ravel()
    unions = np.add.outer(dot_self_X, dot_self_Y) - dot_products

    sims = np.empty_like(dot_products, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            np.log1p(dot_products), np.log1p(unions), out=sims, where=unions >= 1e-8
        )

    sims[unions == 0] = 1
    return sims


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_count_distance(
    X: list | np.ndarray | csr_array, Y: list | np.ndarray | csr_array | None = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 distance for vectors of count values.

    Computes the pairwise Consonni–Todeschini 4 distance between count matrices.
    If one array is passed, distances are computed between its rows. For two arrays,
    distances are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from the first array and `j`-th row from the second array.

    See also :py:func:`ct4_count_distance`.

    Parameters
    ----------
    X : ndarray or CSR sparse array
        First count input array or sparse matrix, of shape :math:`m \times d`.

    Y : ndarray or CSR sparse array, default=None
        Second count input array or sparse matrix, of shape :math:`n \times d`.
        If not passed, distances are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise CT4 distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.
    """
    return 1 - bulk_ct4_count_similarity(X, Y)
