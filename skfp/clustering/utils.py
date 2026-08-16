import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy import sparse


def array_to_bitvectors(X) -> list[ExplicitBitVect]:
    """Convert input data to a list of RDKit ExplicitBitVect objects."""
    if np.ndim(X) == 1 and len(X) > 0 and isinstance(X[0], ExplicitBitVect):
        return list(X)

    bitvecs: list[ExplicitBitVect] = []

    if sparse.issparse(X):
        X = X.tocsr()
        n_samples, n_bits = X.shape
        for i in range(n_samples):
            vec = ExplicitBitVect(n_bits)
            for bit in X.indices[X.indptr[i] : X.indptr[i + 1]]:
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


def assign_labels(
    vectors: list[ExplicitBitVect], centroid_bitvectors: list[ExplicitBitVect]
) -> np.ndarray:
    """Assign each sample to the nearest centroid by Tanimoto similarity."""
    labels = np.empty(len(vectors), dtype=int)
    for i, fp in enumerate(vectors):
        sims = BulkTanimotoSimilarity(fp, centroid_bitvectors)
        labels[i] = int(np.argmax(sims))
    return labels


def clusters_and_points(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Map each cluster ID to the indices of its member samples."""
    return {int(k): np.where(labels == k)[0] for k in np.unique(labels)}
