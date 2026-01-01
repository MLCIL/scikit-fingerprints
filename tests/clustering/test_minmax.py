import pickle
import re

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from rdkit.DataStructs import BulkTanimotoSimilarity

from skfp.clustering import MinMaxClustering


@pytest.fixture
def small_binary_matrix():
    # 6 samples, 8-bit fingerprints
    return np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=int,
    )

def test_fit_clustering_attributes(small_binary_matrix):
    clusterer = MinMaxClustering(distance_threshold=0.5, random_state=42)
    clusterer.fit(small_binary_matrix)
    
    n_samples = small_binary_matrix.shape[0]

    assert hasattr(clusterer, "centroid_indices_")
    assert isinstance(clusterer.centroid_indices_, list)
    assert len(clusterer.centroid_indices_) > 0

    assert hasattr(clusterer, "centroid_bitvectors_")
    assert hasattr(clusterer, "centroids_")

    assert hasattr(clusterer, "labels_")
    assert len(clusterer.labels_) == n_samples
    
def test_assignment_is_nearest_centroid(small_binary_matrix):
    clusterer = MinMaxClustering(distance_threshold=0.5, random_state=42)
    clusterer.fit(small_binary_matrix)

    bitvects = clusterer._array_to_bitvectors(small_binary_matrix)
    centroids = clusterer.centroid_bitvectors_
    labels = clusterer.labels_

    for i, fp in enumerate(bitvects):
        sims = BulkTanimotoSimilarity(fp, centroids)
        assigned_centroid = labels[i]
        max_sim_index = np.argmax(sims)
        assert assigned_centroid == max_sim_index

def test_deterministic_with_fixed_seed(small_binary_matrix):
    c1 = MinMaxClustering(distance_threshold=0.5, random_state=42)
    c2 = MinMaxClustering(distance_threshold=0.5, random_state=42)

    labels1 = c1.fit_predict(small_binary_matrix)
    labels2 = c2.fit_predict(small_binary_matrix)

    assert np.array_equal(labels1, labels2)

def test_empty_input_raises():
    clusterer = MinMaxClustering()
    with pytest.raises(ValueError, match="Empty input"):
        clusterer.fit(np.empty((0, 8)))

def test_invalid_distance_threshold_raises():
    with pytest.raises(ValueError, match="distance_threshold must be between 0 and 1"):
        MinMaxClustering(distance_threshold=1.5)

def test_predict_before_fit_raises(small_binary_matrix):
    clusterer = MinMaxClustering()
    with pytest.raises(
        ValueError,
        match=re.escape("Estimator not fitted. Call fit() first."),
    ):
        clusterer.predict(small_binary_matrix)

def test_sparse_input_handling(small_binary_matrix):
    sparse_matrix = csr_matrix(small_binary_matrix)
    c1 = MinMaxClustering(distance_threshold=0.5, random_state=42)
    c2 = MinMaxClustering(distance_threshold=0.5, random_state=42)

    labels_dense = c1.fit_predict(small_binary_matrix)
    labels_sparse = c2.fit_predict(sparse_matrix)

    assert np.array_equal(labels_dense, labels_sparse)

def test_cloning_works():
    clusterer = MinMaxClustering(distance_threshold=0.5, random_state=42)
  
    cloned_clusterer = clone(clusterer)
    assert cloned_clusterer is not clusterer
    assert isinstance(cloned_clusterer, MinMaxClustering)
    assert cloned_clusterer.distance_threshold == clusterer.distance_threshold
    assert cloned_clusterer.random_state == clusterer.random_state

def test_pickle_roundtrip(small_binary_matrix):
    c = MinMaxClustering(distance_threshold=0.5, random_state=42)
    c.fit(small_binary_matrix)

    blob = pickle.dumps(c)
    c2 = pickle.loads(blob)

    assert np.array_equal(c.labels_, c2.labels_)
    assert c.centroid_indices_ == c2.centroid_indices_
    assert np.array_equal(c.predict(small_binary_matrix),
                           c2.predict(small_binary_matrix))