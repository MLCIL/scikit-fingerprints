import re

import numpy as np
import pytest
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.sparse import csr_matrix

from skfp.clustering import MaxMinClustering


@pytest.fixture(params=["dense", "sparse"])
def binary_X(request):
    X = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    if request.param == "sparse":
        return csr_matrix(X)

    return X


@pytest.fixture(
    params=[
        {},  # using default parameters
        {"distance_threshold": 0.7, "random_state": 42},
    ]
)
def maxmin_clusterer(request):
    return MaxMinClustering(**request.param)


def test_fit_clustering_attributes(maxmin_clusterer, binary_X):
    clusterer = maxmin_clusterer
    clusterer.fit(binary_X)

    n_samples = binary_X.shape[0]

    assert hasattr(clusterer, "centroid_indices_")
    assert isinstance(clusterer.centroid_indices_, list)
    assert len(clusterer.centroid_indices_) > 0

    assert hasattr(clusterer, "centroid_bitvectors_")
    assert hasattr(clusterer, "centroids_")

    assert hasattr(clusterer, "labels_")
    assert len(clusterer.labels_) == n_samples


def test_assignment_is_nearest_centroid(binary_X, maxmin_clusterer):
    clusterer = maxmin_clusterer
    clusterer.fit(binary_X)

    bitvects = clusterer._array_to_bitvectors(binary_X)
    centroids = clusterer.centroid_bitvectors_
    labels = clusterer.labels_

    for i, fp in enumerate(bitvects):
        sims = BulkTanimotoSimilarity(fp, centroids)
        assigned_centroid = labels[i]
        max_sim_index = np.argmax(sims)
        assert assigned_centroid == max_sim_index


def test_deterministic_with_fixed_seed(binary_X):
    c1 = MaxMinClustering(distance_threshold=0.5, random_state=42)
    c2 = MaxMinClustering(distance_threshold=0.5, random_state=42)

    labels1 = c1.fit_predict(binary_X)
    labels2 = c2.fit_predict(binary_X)

    assert np.array_equal(labels1, labels2)


def test_empty_input_raises():
    clusterer = MaxMinClustering()
    with pytest.raises(ValueError, match="Empty input"):
        clusterer.fit(np.empty((0, 8)))


def test_predict_before_fit_raises(binary_X):
    clusterer = MaxMinClustering()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "This MaxMinClustering instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        ),
    ):
        clusterer.predict(binary_X)


def test_sparse_input_handling(binary_X):
    sparse_matrix = csr_matrix(binary_X)
    c1 = MaxMinClustering(distance_threshold=0.5, random_state=42)
    c2 = MaxMinClustering(distance_threshold=0.5, random_state=42)

    labels_dense = c1.fit_predict(binary_X)
    labels_sparse = c2.fit_predict(sparse_matrix)

    assert np.array_equal(labels_dense, labels_sparse)


# def test_cloning_works():
#    clusterer = MaxMinClustering(distance_threshold=0.5, random_state=42)
#
#    cloned_clusterer = clone(clusterer)
#    assert cloned_clusterer is not clusterer
#    assert isinstance(cloned_clusterer, MaxMinClustering)
#    assert cloned_clusterer.distance_threshold == clusterer.distance_threshold
#    assert cloned_clusterer.random_state == clusterer.random_state
#
#
# def test_pickle_roundtrip(binary_X):
#    c = MaxMinClustering(distance_threshold=0.5, random_state=42)
#    c.fit(binary_X)
#
#    blob = pickle.dumps(c)
#    c2 = pickle.loads(blob)
#    assert np.array_equal(c.labels_, c2.labels_)
#    assert c.centroid_indices_ == c2.centroid_indices_
#    assert np.array_equal(
#        c.predict(binary_X), c2.predict(binary_X)
#    )
