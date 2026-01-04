import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.SimDivFilters import MaxMinPicker
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted


class MaxMinClustering(BaseEstimator, ClusterMixin):
    """MaxMin clustering for binary fingerprints using Tanimoto similarity.

    Centroids are selected using RDKit's :class:`~rdkit.SimDivFilters.MaxMinPicker`
    with a distance threshold (distance = 1 - Tanimoto similarity). After
    selecting centroids, each sample is assigned to the centroid with the
    highest Tanimoto similarity.

    Parameters
    ----------
    distance_threshold : float, default=0.5
        Distance threshold used by the MaxMin picker. Must be between 0 and 1.
        Smaller values typically produce more clusters.
    random_state : int or None, default=None
        Seed for centroid selection to ensure deterministic behavior when set.

    Attributes
    ----------
    centroid_indices_ : list of int
        Indices of samples chosen as centroids after :meth:`fit`.
    centroid_bitvectors_ : list of rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Centroid fingerprints as RDKit ExplicitBitVect objects.
    centroids_ : ndarray of bool, shape (n_centroids, n_bits)
        Centroids represented as boolean numpy arrays when input was dense or
        sparse matrix.
    labels_ : ndarray of int, shape (n_samples,)
        Cluster labels for each sample.

    Notes
    -----
    This estimator follows the scikit-learn estimator API and accepts dense
    numpy arrays, scipy sparse matrices, or lists/tuples of RDKit
    :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect` objects as input.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        if not (0.0 <= distance_threshold <= 1.0):
            raise ValueError("Distance_threshold must be between 0 and 1")

        self.distance_threshold = float(distance_threshold)
        self.random_state = None if random_state is None else int(random_state)

    def fit(self, X: np.ndarray | sparse.spmatrix, y=None):
        """Fit the MaxMin clustering model.

        Parameters
        ----------
        X : array-like or sparse matrix or list of ExplicitBitVect
            Binary fingerprint data. Expected shapes are ``(n_samples, n_bits)``
            for arrays and sparse matrices. Alternatively a list/tuple of RDKit
            :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect` objects is
            accepted.
        y : ignored
            Not used, present for API consistency with scikit-learn.

        Returns
        -------
        self : MaxMinClustering
            Fitted estimator with attributes described in the class docstring.

        Raises
        ------
        ValueError
            If `X` is empty or not 2D when provided as an array.
        """
        _ = y  # explicitly unused (sklearn compatibility)

        # Determine number of samples robustly for arrays, lists and sparse matrices
        if sparse.issparse(X):
            n_samples = int(X.shape[0])
        else:
            n_samples = len(X)

        if n_samples == 0:
            raise ValueError("Empty input")

        # --- centroid selection (MaxMin) ---
        picker = MaxMinPicker()

        fps = self._array_to_bitvectors(X)
        seed = -1 if self.random_state is None else int(self.random_state)
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

        # store centroids as boolean numpy arrays
        if sparse.issparse(X) or isinstance(X, np.ndarray):
            arr = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
            self.centroids_ = arr[self.centroid_indices_].astype(bool)

        # --- assignment ---
        self.labels_ = self._assign_labels(fps)

        # enforce invariant: each centroid labels itself
        for cluster_id, sample_idx in enumerate(self.centroid_indices_):
            self.labels_[sample_idx] = cluster_id

        return self

    def _assign_labels(self, bitvecs: list[ExplicitBitVect]) -> np.ndarray:
        """Assign each sample to the nearest centroid by Tanimoto similarity."""
        n_samples = len(bitvecs)
        labels = np.empty(n_samples, dtype=int)

        for i, fp in enumerate(bitvecs):
            sims = BulkTanimotoSimilarity(fp, self.centroid_bitvectors_)
            labels[i] = int(np.argmax(sims))

        return labels

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Assign new samples to existing centroids.

        Parameters
        ----------
        X : array-like or sparse matrix or list of ExplicitBitVect
            New samples to assign to clusters. The input formats match those
            accepted by :meth:`fit`.

        Returns
        -------
        labels : ndarray of int, shape (n_samples,)
            Cluster labels for the input samples.

        Raises
        ------
        ValueError
            If the estimator is not fitted (i.e., ``centroid_bitvectors_`` is
            not present).
        """
        check_is_fitted(self, "centroid_bitvectors_")

        bitvecs = self._array_to_bitvectors(X)
        return self._assign_labels(bitvecs)

    def fit_predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Fit the estimator on ``X`` and return cluster labels.

        This is a convenience method that calls :meth:`fit` followed by returning
        the ``labels_`` attribute.

        Parameters
        ----------
        X : array-like or sparse matrix or list of ExplicitBitVect
            Input data to cluster.

        Returns
        -------
        labels : ndarray of int, shape (n_samples,)
            Cluster labels for ``X``.
        """
        self.fit(X)
        return self.labels_

    def _array_to_bitvectors(
        self, X: np.ndarray | sparse.spmatrix
    ) -> list[ExplicitBitVect]:
        # Case 1: already RDKit fingerprints
        if isinstance(X, (list, tuple)) and isinstance(X[0], ExplicitBitVect):
            return list(X)
        # Case 2: sparse matrix
        if sparse.issparse(X):
            X = X.tocoo()
            n_samples, n_bits = X.shape
            bitvecs = [ExplicitBitVect(n_bits) for _ in range(n_samples)]
            for i, j, v in zip(X.row, X.col, X.data, strict=True):
                if v:
                    bitvecs[i].SetBit(int(j))
            return bitvecs

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_bits)")

        n_samples, n_bits = arr.shape
        bitvecs = [ExplicitBitVect(n_bits) for _ in range(n_samples)]
        for i in range(n_samples):
            for bit in np.nonzero(arr[i])[0]:
                bitvecs[i].SetBit(int(bit))
        return bitvecs

    def get_clusters(self) -> dict[int, np.ndarray]:
        """Return clusters as a mapping from cluster id to sample indices.

        Returns
        -------
        dict
            Mapping from integer cluster id to a 1D numpy array containing the
            indices of samples belonging to that cluster.

        Raises
        ------
        ValueError
            If the estimator is not fitted (``labels_`` missing).
        """
        check_is_fitted(self, "labels_")
        return {
            k: np.where(self.labels_ == k)[0]
            for k in range(len(self.centroid_indices_))
        }
