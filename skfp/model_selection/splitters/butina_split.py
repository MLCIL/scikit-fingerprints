from collections.abc import Sequence
from numbers import Integral
from typing import Any

from rdkit.Chem import Mol
from rdkit.ML.Cluster import Butina
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.distances import bulk_tanimoto_binary_distance
from skfp.fingerprints import ECFPFingerprint
from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices
from skfp.utils.validators import ensure_mols


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "threshold": [Interval(RealNotInt, 0, 1, closed="both")],
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def butina_train_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    test_size: float | None = None,
    threshold: float = 0.65,
    return_indices: bool = False,
) -> (
    tuple[Sequence[str | Mol], Sequence[str | Mol], Sequence[Sequence[Any]]]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int]]
):
    """
    Split using Taylor-Butina clustering.

    This split uses deterministically partitioned clusters of molecules from Taylor-Butina
    clustering [1]_ [2]_ [3]_. It aims to verify the model generalization to structurally
    novel molecules. Also known as sphere exclusion or leader-following clustering.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle [4]_ for RDKit. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset and the rest to the training subset.

    If ``train_size`` and ``test_size`` are integers, they must sum up to the ``data``
    length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

    threshold : float, default=0.65
        Tanimoto distance threshold, defining the minimal distance between cluster
        centroids. Default value is based on ECFP4 activity threshold as determined
        by Roger Sayle [4]_.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If `return_indices`
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Darko Butina
        "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
        Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets"
        . Chem. Inf. Comput. Sci. 1999, 39, 4, 747-750
        <https://pubs.acs.org/doi/abs/10.1021/ci9803381>`_

    .. [2] `Robin Taylor
        "Simulation Analysis of Experimental Design Strategies for Screening Random
        Compounds as Potential New Drugs and Agrochemicals"
        J. Chem. Inf. Comput. Sci. 1995, 35, 1, 59-67
        <https://pubs.acs.org/doi/10.1021/ci00023a009>`_

    .. [3] `Noel O'Boyle
        "Taylor-Butina Clustering"
        <https://noel.redbrick.dcu.ie/R_clustering.html>`_

    .. [4] `Roger Sayle
        "2D similarity, diversity and clustering in RDKit"
        RDKit UGM 2019
        <https://www.nextmovesoftware.com/talks/Sayle_2DSimilarityDiversityAndClusteringInRdkit_RDKITUGM_201909.pdf>`_

    Examples
    --------
    >>> from skfp.model_selection.splitters import butina_train_test_split
    >>> smiles = ['c1ccccc1', 'c1cccnc1', 'c1ccncc1', 'CC(=O)O', 'CC(N)=O', 'CCCCCC', 'CCCCCN', 'c1ccc(O)cc1']
    >>> train_smiles, test_smiles = butina_train_test_split(smiles, train_size=0.75, test_size=0.25)
    >>> train_smiles
    ['CCCCCN', 'CCCCCC', 'CC(N)=O', 'CC(=O)O', 'c1ccncc1', 'c1cccnc1']
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )

    clusters = _create_clusters(data, threshold)
    clusters.sort(key=len)

    train_idxs: list[int] = []
    test_idxs: list[int] = []

    for cluster in clusters:
        if len(test_idxs) < test_size:
            test_idxs.extend(cluster)
        else:
            train_idxs.extend(cluster)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return train_subset, test_subset, *additional_data_split
    else:
        return train_subset, test_subset


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "threshold": [Interval(RealNotInt, 0, 1, closed="both")],
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def butina_train_valid_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    threshold: float = 0.65,
    return_indices: bool = False,
) -> (
    tuple[
        Sequence[str | Mol],
        Sequence[str | Mol],
        Sequence[str | Mol],
        Sequence[Sequence[Any]],
    ]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int], Sequence[int]]
):
    """
    Split using Taylor-Butina clustering.

    This split uses deterministically partitioned clusters of molecules from Taylor-Butina
    clustering [1]_ [2]_ [3]_. It aims to verify the model generalization to structurally
    novel molecules.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. They are then clustered using Leader Clustering, a variant of Taylor-Butina
    clustering by Roger Sayle for RDKit [4]_. Cluster centroids (central molecules) are
    guaranteed to have at least a given Tanimoto distance between them, as defined by
    `threshold` parameter.

    Clusters are divided deterministically, with the smallest clusters assigned to the
    test subset, larger to the validation subset, and the rest to the training subset

    If ``train_size``, ``valid_size`` and ``test_size`` are integers, they must sum up
    to the ``data`` length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data, e.g. labels.

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set
        to 1 - test_size - valid_size. If valid_size is not provided, train_size
        is set to 1 - test_size. If train_size, test_size and valid_size aren't
        set, train_size is set to 0.8.

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set
        to 1 - train_size - valid_size. If train_size, test_size and valid_size
        aren't set, train_size is set to 0.1.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is
        set to 1 - train_size - valid_size. If valid_size is not provided, test_size
        is set to 1 - train_size. If train_size, test_size and valid_size aren't set,
        test_size is set to 0.1.

    threshold : float, default=0.65
        Tanimoto distance threshold, defining the minimal distance between cluster
        centroids. Default value is based on ECFP4 activity threshold as determined
        by Roger Sayle [4]_.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If
        `return_indices` is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Darko Butina
        "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
        Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets"
        . Chem. Inf. Comput. Sci. 1999, 39, 4, 747-750
        <https://pubs.acs.org/doi/abs/10.1021/ci9803381>`_

    .. [2] `Robin Taylor
        "Simulation Analysis of Experimental Design Strategies for Screening Random
        Compounds as Potential New Drugs and Agrochemicals"
        J. Chem. Inf. Comput. Sci. 1995, 35, 1, 59-67
        <https://pubs.acs.org/doi/10.1021/ci00023a009>`_

    .. [3] `Noel O'Boyle "Taylor-Butina Clustering"
        <https://noel.redbrick.dcu.ie/R_clustering.html>`_

    .. [4] `Roger Sayle
        "2D similarity, diversity and clustering in RDKit"
        RDKit UGM 2019
        <https://www.nextmovesoftware.com/talks/Sayle_2DSimilarityDiversityAndClusteringInRdkit_RDKITUGM_201909.pdf>`_

    Examples
    --------
    >>> from skfp.model_selection.splitters import butina_train_valid_test_split
    >>> smiles = ['c1ccccc1', 'c1cccnc1', 'c1ccncc1', 'CC(=O)O', 'CC(N)=O', 'CCCCCC', 'CCCCCN', 'c1ccc(O)cc1']
    >>> train_smiles, valid_smiles, test_smiles = butina_train_valid_test_split(
    ...     smiles, train_size=0.5, valid_size=0.25, test_size=0.25
    ... )
    >>> train_smiles
    ['CC(N)=O', 'CC(=O)O', 'c1ccncc1', 'c1cccnc1']
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    clusters = _create_clusters(data, threshold)
    clusters.sort(key=len)

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []

    for cluster in clusters:
        if len(test_idxs) < test_size:
            test_idxs.extend(cluster)
        elif len(valid_idxs) < valid_size:
            valid_idxs.extend(cluster)
        else:
            train_idxs.extend(cluster)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(valid_idxs, "validation")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_clusters(
    data: Sequence[str | Mol], threshold: float = 0.65
) -> list[tuple[int]]:
    """
    Generate Taylor-Butina clusters for a list of SMILES strings or RDKit ``Mol`` objects.
    This function groups molecules by using clustering, where cluster centers must have
    Tanimoto (Jaccard) distance greater or equal to a given threshold. Binary ECFP4 (Morgan)
    fingerprints with 2048 bits are used as features.
    """
    mols = ensure_mols(data)

    fps = ECFPFingerprint().transform(mols)
    dists = bulk_tanimoto_binary_distance(fps)

    clusters = Butina.ClusterData(
        dists,
        nPts=len(fps),
        distThresh=threshold,
        isDistData=True,
        reordering=True,
    )

    return list(clusters)
