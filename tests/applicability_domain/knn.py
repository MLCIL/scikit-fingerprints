import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.applicability_domain import KNNADChecker
from tests.applicability_domain.utils import (
    get_data_inside_ad,
    get_data_outside_ad,
    get_metric_data_type,
)

ALLOWED_METRICS = [
    "tanimoto_binary_distance",
    "tanimoto_count_distance",
]

ALLOWED_AGGREGATIONS = ["mean", "max", "min"]


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGREGATIONS)
def test_inside_knn_ad(metric, agg):
    X_train, X_test = get_data_inside_ad(
        n_train=100, n_test=10, data_type=get_metric_data_type(metric)
    )

    k = 1 if agg == "min" else 3
    ad_checker = KNNADChecker(k=k, metric=metric, agg=agg)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert_equal(scores.shape, (len(X_test),))
    assert scores.shape == (len(X_test),)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGREGATIONS)
def test_outside_knn_ad(metric, agg):
    X_train, X_test = get_data_outside_ad(
        n_train=100, n_test=10, data_type=get_metric_data_type(metric)
    )

    k = 1 if agg == "min" else 3
    ad_checker = KNNADChecker(k=k, metric=metric, agg=agg)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGREGATIONS)
def test_knn_different_k_values(metric, agg):
    X_train, X_test = get_data_inside_ad(
        n_train=100, n_test=10, data_type=get_metric_data_type(metric)
    )

    # smaller k, stricter check
    ad_checker_k1 = KNNADChecker(k=1, metric=metric, agg=agg)
    ad_checker_k1.fit(X_train)
    passed_k1 = ad_checker_k1.predict(X_test).sum()

    # larger k, potentially less strict
    ad_checker_k5 = KNNADChecker(k=5, metric=metric, agg=agg)
    ad_checker_k5.fit(X_train)
    passed_k5 = ad_checker_k5.predict(X_test).sum()

    # both should be valid results
    assert isinstance(passed_k1, (int, np.integer))
    assert isinstance(passed_k5, (int, np.integer))


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGREGATIONS)
def test_knn_pass_y_train(metric, agg):
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad(
        n_train=100, n_test=10, data_type=get_metric_data_type(metric)
    )

    y_train = np.zeros(len(X_train))
    k = 1 if agg == "min" else 3
    ad_checker = KNNADChecker(k=k, metric=metric, agg=agg)
    ad_checker.fit(X_train, y_train)


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGREGATIONS)
def test_knn_invalid_k(metric, agg):
    X_train, _ = get_data_inside_ad(
        n_train=100, n_test=10, data_type=get_metric_data_type(metric)
    )

    with pytest.raises(
        ValueError,
        match=r"k \(\d+\) must be smaller than or equal to the number of training samples \(\d+\)",
    ):
        ad_checker = KNNADChecker(k=len(X_train) + 1, metric=metric, agg=agg)
        ad_checker.fit(X_train)


def test_knn_min_agg_with_k_above_1_warns():
    X_train, _ = get_data_inside_ad(n_train=100)

    ad_checker = KNNADChecker(k=3, agg="min")
    with pytest.warns(UserWarning, match=r"k should be 1 when agg='min'"):
        ad_checker.fit(X_train)


def test_knn_invalid_metric():
    X_train, _ = get_data_inside_ad(n_train=100)
    ad_checker = KNNADChecker(k=3, metric="euclidean")
    with pytest.raises(KeyError):
        ad_checker.fit(X_train)
