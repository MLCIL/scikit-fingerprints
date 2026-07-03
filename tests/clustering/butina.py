import re

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from skfp.clustering import ButinaClustering


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
        {},  # default parameters
        {"distance_threshold": 0.7},
    ]
)
def butina_clusterer(request):
    return ButinaClustering(**request.param)


def test_fit_clustering_attributes(butina_clusterer, binary_X):
    clusterer = butina_clusterer.fit(binary_X)
    n_samples = binary_X.shape[0]

    assert hasattr(clusterer, "centroid_indices_")
    assert isinstance(clusterer.centroid_indices_, list)
    assert len(clusterer.centroid_indices_) > 0

    assert hasattr(clusterer, "centroid_bitvectors_")
    assert hasattr(clusterer, "centroids_")

    assert hasattr(clusterer, "labels_")
    assert len(clusterer.labels_) == n_samples


def test_labels_within_cluster_range(butina_clusterer, binary_X):
    clusterer = butina_clusterer.fit(binary_X)
    n_clusters = len(clusterer.centroid_indices_)
    assert clusterer.labels_.min() >= 0
    assert clusterer.labels_.max() < n_clusters
    # every cluster id is used
    assert set(clusterer.labels_.tolist()) == set(range(n_clusters))


def test_deterministic(binary_X):
    c1 = ButinaClustering(distance_threshold=0.5)
    c2 = ButinaClustering(distance_threshold=0.5)
    assert np.array_equal(c1.fit_predict(binary_X), c2.fit_predict(binary_X))


def test_sparse_matches_dense(binary_X):
    dense = np.asarray(binary_X.todense()) if hasattr(binary_X, "todense") else binary_X
    labels_dense = ButinaClustering(distance_threshold=0.5).fit(dense).labels_
    labels_sparse = (
        ButinaClustering(distance_threshold=0.5).fit(csr_matrix(dense)).labels_
    )
    assert np.array_equal(labels_dense, labels_sparse)


def test_predict_assigns_to_nearest_centroid(binary_X):
    from rdkit.DataStructs import BulkTanimotoSimilarity

    clusterer = ButinaClustering(distance_threshold=0.5).fit(binary_X)
    bitvects = clusterer._array_to_bitvectors(binary_X)
    preds = clusterer.predict(binary_X)
    for i, fp in enumerate(bitvects):
        sims = BulkTanimotoSimilarity(fp, clusterer.centroid_bitvectors_)
        assert preds[i] == int(np.argmax(sims))


def test_get_clusters_and_points(binary_X):
    clusterer = ButinaClustering(distance_threshold=0.5).fit(binary_X)
    clusters = clusterer.get_clusters_and_points()
    for k, idx in clusters.items():
        assert np.all(clusterer.labels_[idx] == k)


def test_empty_input_raises():
    clusterer = ButinaClustering()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Found array with 0 sample(s) (shape=(0, 8)) while a minimum of 1 is "
            "required by ButinaClustering."
        ),
    ):
        clusterer.fit(np.empty((0, 8)))


def test_predict_before_fit_raises(binary_X):
    clusterer = ButinaClustering()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "This ButinaClustering instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        ),
    ):
        clusterer.predict(binary_X)
