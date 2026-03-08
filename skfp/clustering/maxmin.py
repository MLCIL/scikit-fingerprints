from collections.abc import Sequence

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.SimDivFilters import MaxMinPicker
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted, validate_data


class MaxMinClustering(BaseEstimator, ClusterMixin):
    """
    MaxMin clustering.

    This is a centroid-based clustering algorithm using binary fingerprints
    and Tanimoto similarity. Centroids are selected to maximize their minimal
    pairwise distance, distributing them uniformly across the space. Clusters
    tend to be convex and similarly sized, in contrast to density-based clustering
    methods like Butina clustering.

    Centroid selection uses the MaxMin heuristic originally described by
    Ashton et al. [1]_, where they are iteratively selected to maximize the
    minimal distance to previously chosen centroids. RDKit ``MaxMinPicker`` with
    a given minimal distance threshold is used.

    After selecting centroids, each sample is assigned to the centroid with the highest
    Tanimoto similarity. The same process is used for prediction for new samples.

    Parameters
    ----------
    distance_threshold : float, default=0.1
        Distance threshold, denotes minimal Tanimoto distance between clusters
        (distance = 1 - Tanimoto similarity) Must be between 0 and 1.
        The default value was chosen based on analysis of multiple chemical datasets [2]_.

    random_state : int, RandomState instance or None, default=None
       Determines random number generation for selection of the first centroid.
       Pass an integer for reproducible output across multiple function calls.

    Attributes
    ----------
    centroid_indices_ : list of int
        Indices of samples chosen as centroids after :meth:`fit`.

    centroid_bitvectors_ : list of ExplicitBitVect
        Centroid fingerprints as RDKit ExplicitBitVect objects.

    centroids_ : ndarray of bool, shape (n_centroids, n_bits)
        Centroids represented as boolean NumPy arrays when the input was a
        dense array or sparse matrix.

    labels_ : ndarray of int, shape (n_samples,)
        Cluster labels for each sample.

    Notes
    -----
    This estimator follows the scikit-learn estimator API and accepts dense
    NumPy arrays, SciPy sparse arrays, or lists/tuples of RDKit
    ``ExplicitBitVect`` objects as input.

    References
    ----------
    .. [1] `Ashton, M., Barnard, J. M., Casset, F., Charlton, M. H., Downs, G. M.,
        and Willett, P.
        "Identification of Diverse Database Subsets using Property-Based and
        Fragment-Based Molecular Descriptions"
        Quant. Struct.-Act. Relat., 2002, 21: 598-604
        <https://doi.org/10.1002/qsar.200290002>`_

    .. [2] `Sayle, R. A.
        "2D similarity, diversity, and clustering in RDKit"
        RDKit User Group Meeting 2019
        <https://github.com/rdkit/UGM_2019/blob/master/Presentations/Sayle_Clustering.pdf>`_
    """

    _parameter_constraints: dict = {
        "distance_threshold": [Interval(RealNotInt, 0, 1, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        distance_threshold: float = 0.1,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.distance_threshold = distance_threshold
        self.random_state = random_state

    def fit(self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect], y=None):  # noqa: ARG002
        """
        Fit the MaxMin clustering model.

        Parameters
        ----------
        X : {array-like, sparse matrix, sequence of ExplicitBitVect}
            Binary fingerprint data. Expected shapes are ``(n_samples, n_bits)``
            for arrays and sparse arrays. Alternatively, a list/tuple of RDKit
            ``ExplicitBitVect`` objects is accepted.

        y : ignored
            Not used, present for API consistency with scikit-learn.

        Returns
        -------
        self : MaxMinClustering
            Fitted estimator.
        """
        super()._validate_params()
        X = validate_data(self, X, accept_sparse=["csr"], ensure_2d=False)

        # centroid selection (MaxMin)
        picker = MaxMinPicker()

        fps = self._array_to_bitvectors(X)
        rng = check_random_state(self.random_state)
        seed = rng.randint(0, 2**31 - 1)
        centroid_indices, _ = picker.LazyBitVectorPickWithThreshold(
            fps,
            poolSize=len(fps),
            pickSize=len(fps),
            threshold=self.distance_threshold,
            seed=seed,
        )
        centroid_indices = list(centroid_indices)

        self.centroid_indices_ = centroid_indices
        self.centroid_bitvectors_ = [fps[i] for i in centroid_indices]

        # store centroids as NumPy arrays
        if sparse.issparse(X) or isinstance(X, np.ndarray):
            arr = X.todense() if sparse.issparse(X) else X
            self.centroids_ = arr[self.centroid_indices_].astype(np.uint8)

        # cluster assignment
        self.labels_ = self._assign_labels(fps)

        # enforce invariant: each centroid labels itself
        for cluster_id, sample_idx in enumerate(self.centroid_indices_):
            self.labels_[sample_idx] = cluster_id

        return self

    def predict(
        self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect]
    ) -> np.ndarray:
        """
        Assign new samples to existing centroids.

        Parameters
        ----------
        X : {array-like, sparse matrix, sequence of ExplicitBitVect}
            New samples to assign to clusters. The input formats match those
            accepted by :meth:`fit`.

        Returns
        -------
        labels : ndarray of int, shape (n_samples,)
            Cluster labels for the input samples.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=["csr"], ensure_2d=False)

        bitvecs = self._array_to_bitvectors(X)
        return self._assign_labels(bitvecs)

    def fit_predict(
        self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect], y=None  # noqa: ARG002
    ) -> np.ndarray:
        """
        Fit the MaxMin clustering model and return cluster labels.

        This is a convenience method that calls :meth:`fit` and returns the
        cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix, sequence of ExplicitBitVect}
            Binary fingerprint data. Expected shapes are ``(n_samples, n_bits)``
            for arrays and sparse arrays. Alternatively, a list/tuple of RDKit
            ``ExplicitBitVect`` objects is accepted.

        y : ignored
            Not used, present for API consistency with scikit-learn.

        Returns
        -------
        labels : ndarray of int, shape (n_samples,)
            Cluster labels for ``X``.
        """
        self.fit(X)
        return self.labels_

    def get_clusters_and_points(self) -> dict[int, np.ndarray]:
        """
        Return clusters as a mapping from cluster ID to sample indices.

        Returns
        -------
        clusters : dict
            Mapping from integer cluster ID to a 1D NumPy array containing the
            indices of samples belonging to that cluster.
        """
        check_is_fitted(self)
        return {
            k: np.where(self.labels_ == k)[0]
            for k in range(len(self.centroid_indices_))
        }

    def _array_to_bitvectors(
        self, X: np.ndarray | sparse.csr_array
    ) -> list[ExplicitBitVect]:
        """
        Convert input data to a list of RDKit ExplicitBitVect objects.
        """
        bitvecs: list[ExplicitBitVect] = []
        if np.ndim(X) == 1 and len(X) > 0 and isinstance(X[0], ExplicitBitVect):
            return list(X)

        if sparse.issparse(X):
            X = X.tocsr()
            n_samples, n_bits = X.shape

            for i in range(n_samples):
                vec = ExplicitBitVect(n_bits)
                row_start = X.indptr[i]
                row_end = X.indptr[i + 1]

                for bit in X.indices[row_start:row_end]:
                    # RDKit ExplicitBitVect uses int indices, not Numpy integers
                    vec.SetBit(int(bit))

                bitvecs.append(vec)

            return bitvecs

        n_samples, n_bits = X.shape

        for i in range(n_samples):
            vec = ExplicitBitVect(n_bits)
            for bit in np.flatnonzero(X[i]):
                vec.SetBit(int(bit))
            bitvecs.append(vec)

        return bitvecs

    def _assign_labels(self, vectors: list[ExplicitBitVect]) -> np.ndarray:
        """
        Assign each sample to the nearest centroid by Tanimoto similarity.
        """
        n_samples = len(vectors)
        labels = np.empty(n_samples, dtype=int)

        for i, fp in enumerate(vectors):
            sims = BulkTanimotoSimilarity(fp, self.centroid_bitvectors_)
            labels[i] = np.argmax(sims)

        return labels
