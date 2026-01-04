import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.SimDivFilters import MaxMinPicker
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted


class MinMaxClustering(BaseEstimator, ClusterMixin):
    """
    MinMax clustering for binary fingerprints using Tanimoto similarity.

    Centroids are selected using RDKit's MaxMinPicker with a distance threshold
    (distance = 1 - Tanimoto similarity). All samples are then assigned to the
    nearest centroid by maximum Tanimoto similarity.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        if not (0.0 <= distance_threshold <= 1.0):
            raise ValueError("distance_threshold must be between 0 and 1")

        self.distance_threshold = float(distance_threshold)
        self.random_state = None if random_state is None else int(random_state)

    def fit(self, X: np.ndarray | sparse.spmatrix, y=None):
        """Fit the MinMax clustering model."""
        _ = y  # explicitly unused (sklearn compatibility)

        # Determine number of samples robustly for arrays, lists and sparse matrices
        if sparse.issparse(X):
            n_samples = int(X.shape[0])
        else:
            n_samples = len(X)

        if n_samples == 0:
            raise ValueError("Empty input")

        # --- centroid selection (MinMax) ---
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
        """Assign new samples to existing centroids."""
        check_is_fitted(self, "centroid_bitvectors_")

        bitvecs = self._array_to_bitvectors(X)
        return self._assign_labels(bitvecs)

    def fit_predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Fit and return cluster labels for X."""
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
        """
        Get the clusters as a dictionary mapping cluster IDs to arrays of sample indices.
        """
        check_is_fitted(self, "labels_")
        return {
            k: np.where(self.labels_ == k)[0]
            for k in range(len(self.centroid_indices_))
        }
