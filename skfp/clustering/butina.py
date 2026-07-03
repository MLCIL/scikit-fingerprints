from collections.abc import Sequence

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted, validate_data


class ButinaClustering(BaseEstimator, ClusterMixin):
    """
    Taylor-Butina clustering.

    A density-based clustering algorithm for binary fingerprints using Tanimoto
    similarity, also known as sphere exclusion or leader-following clustering [1]_ [2]_.
    Cluster centroids are chosen so that no two centroids are closer than a given
    Tanimoto distance ``distance_threshold``; every remaining sample joins the cluster
    of the first centroid within that radius. In contrast to centroid-based methods
    like MaxMin clustering, Butina clusters follow the density of the data and can vary
    widely in size.

    Clustering uses RDKit ``Butina.ClusterData`` with ``reordering=True`` for
    deterministic output. After fitting, new samples are assigned with :meth:`predict`
    to the centroid with the highest Tanimoto similarity.

    Parameters
    ----------
    distance_threshold : float, default=0.65
        Tanimoto distance threshold (distance = 1 - Tanimoto similarity), the minimal
        distance between cluster centroids. Must be between 0 and 1. The default follows
        the ECFP4 activity threshold used by the Butina train/test splitter [3]_.

    Attributes
    ----------
    centroid_indices_ : list of int
        Indices of samples chosen as cluster centroids after :meth:`fit`.

    centroid_bitvectors_ : list of ExplicitBitVect
        Centroid fingerprints as RDKit ExplicitBitVect objects.

    centroids_ : ndarray of uint8, shape (n_centroids, n_bits)
        Centroids as a boolean NumPy array when the input was a dense array or
        sparse matrix.

    labels_ : ndarray of int, shape (n_samples,)
        Cluster labels for each sample.

    Notes
    -----
    This estimator follows the scikit-learn estimator API and accepts dense NumPy
    arrays, SciPy CSR sparse arrays, or lists/tuples of RDKit ``ExplicitBitVect``
    objects as input.

    References
    ----------
    .. [1] `Darko Butina
        "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto
        Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets"
        J. Chem. Inf. Comput. Sci., 1999, 39, 4, 747-750
        <https://doi.org/10.1021/ci9803381>`_

    .. [2] `Robin Taylor
        "Simulation Analysis of Experimental Design Strategies for Screening Random
        Compounds as Potential New Drugs and Agrochemicals"
        J. Chem. Inf. Comput. Sci., 1995, 35, 1, 59-67
        <https://doi.org/10.1021/ci00023a009>`_

    .. [3] `Roger A. Sayle
        "2D similarity, diversity and clustering in RDKit"
        RDKit User Group Meeting 2019
        <https://github.com/rdkit/UGM_2019/blob/master/Presentations/Sayle_Clustering.pdf>`_
    """

    _parameter_constraints: dict = {
        "distance_threshold": [Interval(RealNotInt, 0, 1, closed="both")],
    }

    def __init__(self, distance_threshold: float = 0.65):
        self.distance_threshold = distance_threshold

    def fit(self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect], y=None):  # noqa: ARG002
        """
        Fit the Butina clustering model.

        Parameters
        ----------
        X : {array-like, sparse matrix, sequence of ExplicitBitVect}
            Binary fingerprint data of shape ``(n_samples, n_bits)``, or a
            list/tuple of RDKit ``ExplicitBitVect`` objects.

        y : ignored
            Not used, present for API consistency with scikit-learn.

        Returns
        -------
        self : ButinaClustering
            Fitted estimator.
        """
        super()._validate_params()
        X = validate_data(self, X, accept_sparse=["csr"], ensure_2d=False)

        fps = self._array_to_bitvectors(X)
        n_samples = len(fps)

        # Condensed lower-triangle Tanimoto distances, the format expected by
        # Butina.ClusterData(isDistData=True): dist(i, j) for i > j, row-major.
        condensed_distances: list[float] = []
        for i in range(1, n_samples):
            sims = BulkTanimotoSimilarity(fps[i], fps[:i])
            condensed_distances.extend(1.0 - sim for sim in sims)

        clusters = Butina.ClusterData(
            condensed_distances,
            n_samples,
            self.distance_threshold,
            isDistData=True,
            reordering=True,
        )

        # In each RDKit Butina cluster the first element is the centroid.
        self.centroid_indices_ = [cluster[0] for cluster in clusters]
        self.centroid_bitvectors_ = [fps[i] for i in self.centroid_indices_]

        if sparse.issparse(X) or isinstance(X, np.ndarray):
            arr = X.todense() if sparse.issparse(X) else X
            self.centroids_ = np.asarray(arr)[self.centroid_indices_].astype(np.uint8)

        labels = np.empty(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_id
        self.labels_ = labels

        return self

    def predict(
        self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect]
    ) -> np.ndarray:
        """
        Assign new samples to existing centroids.

        Parameters
        ----------
        X : {array-like, sparse matrix, sequence of ExplicitBitVect}
            New samples to assign. The input formats match those accepted by :meth:`fit`.

        Returns
        -------
        labels : ndarray of int, shape (n_samples,)
            Cluster labels for the input samples, assigned to the nearest centroid by
            Tanimoto similarity.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=["csr"], ensure_2d=False)
        bitvecs = self._array_to_bitvectors(X)
        return self._assign_labels(bitvecs)

    def fit_predict(
        self, X: np.ndarray | sparse.csr_array | Sequence[ExplicitBitVect], y=None
    ) -> np.ndarray:
        """
        Fit the Butina clustering model and return cluster labels.

        This is a convenience method that calls :meth:`fit` and returns ``labels_``.
        """
        self.fit(X, y)
        return self.labels_

    def get_clusters_and_points(self) -> dict[int, np.ndarray]:
        """
        Return a mapping from cluster id to the indices of its member samples.
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
        if np.ndim(X) == 1 and len(X) > 0 and isinstance(X[0], ExplicitBitVect):
            return list(X)

        bitvecs: list[ExplicitBitVect] = []

        if sparse.issparse(X):
            X = X.tocsr()
            n_samples, n_bits = X.shape
            for i in range(n_samples):
                vec = ExplicitBitVect(n_bits)
                row_start, row_end = X.indptr[i], X.indptr[i + 1]
                for bit in X.indices[row_start:row_end]:
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
        labels = np.empty(len(vectors), dtype=int)
        for i, fp in enumerate(vectors):
            sims = BulkTanimotoSimilarity(fp, self.centroid_bitvectors_)
            labels[i] = int(np.argmax(sims))
        return labels
