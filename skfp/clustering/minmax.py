from typing import List
import numpy as np
from scipy import sparse
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from sklearn.base import ClusterMixin


class BaseClusterer(ClusterMixin):
    """Base class for clustering with binary fingerprints."""

    def _array_to_bitvectors(
        self, X: np.ndarray | sparse.spmatrix
    ) -> List[ExplicitBitVect]:
        if sparse.issparse(X):
            X = X.tocoo()
            n_samples, n_bits = X.shape
            bitvecs = [ExplicitBitVect(n_bits) for _ in range(n_samples)]
            for i, j, v in zip(X.row, X.col, X.data):
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


class MinMaxClustering(BaseClusterer):
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

    def fit(self, X: np.ndarray | sparse.spmatrix):
        """Fit the MinMax clustering model."""
        bitvecs = self._array_to_bitvectors(X)
        n_samples = len(bitvecs)
        if n_samples == 0:
            raise ValueError("Empty input")

        # --- centroid selection (MinMax) ---
        picker = MaxMinPicker()
        centroid_indices, _ = picker.LazyBitVectorPickWithThreshold(
            bitvecs,
            poolSize=n_samples,
            pickSize=n_samples,
            threshold=self.distance_threshold,
            seed=self.random_state,
        )
        centroid_indices = list(centroid_indices)

        self.centroid_indices_ = centroid_indices
        self.centroid_bitvectors_ = [bitvecs[i] for i in centroid_indices]

        # store centroids as boolean numpy arrays
        arr = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
        self.centroids_ = arr[centroid_indices].astype(bool)

        # --- assignment ---
        self.labels_ = self._assign_labels(bitvecs)

        return self

    def _assign_labels(self, bitvecs: List[ExplicitBitVect]) -> np.ndarray:
        """Assign each sample to the nearest centroid by Tanimoto similarity."""
        n_samples = len(bitvecs)
        labels = np.empty(n_samples, dtype=int)

        for i, fp in enumerate(bitvecs):
            sims = BulkTanimotoSimilarity(fp, self.centroid_bitvectors_)
            labels[i] = int(np.argmax(sims))

        return labels

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Assign new samples to existing centroids."""
        if not hasattr(self, "centroid_bitvectors_"):
            raise ValueError("Estimator not fitted. Call fit() first.")

        bitvecs = self._array_to_bitvectors(X)
        return self._assign_labels(bitvecs)

    def fit_predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Fit and return cluster labels for X."""
        self.fit(X)
        return self.labels_
