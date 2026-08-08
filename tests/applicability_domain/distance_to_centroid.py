import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.applicability_domain import DistanceToCentroidADChecker
from skfp.applicability_domain.distance_to_centroid import SCIPY_METRIC_NAMES
from skfp.distances import (
    _BULK_METRIC_NAMES as SKFP_BULK_METRIC_NAMES,
)
from skfp.distances import (
    _METRIC_NAMES as SKFP_METRIC_NAMES,
)
from skfp.distances import (
    _METRICS as SKFP_METRICS,
)
from tests.applicability_domain.utils import (
    get_data_inside_ad,
    get_data_outside_ad,
    get_metric_data_type,
)


def test_inside_distance_to_centroid_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_outside_distance_to_centroid_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


def test_distance_to_centroid_lower_distance():
    X_train, _ = get_data_inside_ad()
    X_test = X_train + 10

    # looser check, AD is larger
    ad_checker = DistanceToCentroidADChecker(threshold=100)
    ad_checker.fit(X_train)
    passed_dist_100 = ad_checker.predict(X_test).sum()

    # stricter check, AD is smaller
    ad_checker = DistanceToCentroidADChecker(threshold=10)
    ad_checker.fit(X_train)
    passed_dist_10 = ad_checker.predict(X_test).sum()

    # larger distance = larger AD = more passed points
    assert passed_dist_100 > passed_dist_10


@pytest.mark.parametrize(
    "metric",
    list(SCIPY_METRIC_NAMES)
    + list(SKFP_BULK_METRIC_NAMES)
    + list(SKFP_METRIC_NAMES)
    + list(SKFP_METRICS.values()),
)
def test_distance_to_centroid_different_metrics(metric):
    X_train, X_test = get_data_inside_ad(data_type=get_metric_data_type(metric))

    # smoke test, should not throw errors
    ad_checker = DistanceToCentroidADChecker(metric=metric)
    ad_checker.fit(X_train)
    ad_checker.predict(X_test)


def test_distance_to_centroid_pass_y_train():
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad()
    y_train = np.zeros(len(X_train))
    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train, y_train)
