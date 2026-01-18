import inspect

import numpy as np
import pytest
from sklearn.utils.estimator_checks import (
    check_dont_overwrite_parameters,
    check_estimators_fit_returns_self,
    check_estimators_pickle,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_set_params,
)

import skfp.clustering

# global test data (simple, estimator-agnostic)
n_samples = 10
rng = np.random.RandomState(0)
X = rng.randint(0, 2, size=(n_samples, 8), dtype=np.uint8)
y = None


def get_all_clusterers():
    return [
        (name, obj)
        for name, obj in inspect.getmembers(skfp.clustering)
        if inspect.isclass(obj)
    ]


@pytest.mark.parametrize("name, cls", get_all_clusterers())
def test_basic_sklearn_checks_for_clusterers(name, cls):
    estimator = cls()
    run_clusterer_checks(name, estimator)


def run_clusterer_checks(name, estimator):
    # checks that only need (name, estimator)
    name_estimator_checks = [
        check_no_attributes_set_in_init,
        check_get_params_invariance,
        check_set_params,
        check_dont_overwrite_parameters,
    ]

    # checks that need (estimator, X, y)
    data_checks = [
        check_estimators_fit_returns_self,
        check_estimators_pickle,
    ]

    for check in name_estimator_checks:
        try:
            check(name, estimator)
        except Exception:
            print(f"\n{name} failed check {check.__name__}")
            raise

    for check in data_checks:
        try:
            check(estimator, X, y)
        except Exception:
            print(f"\n{name} failed check {check.__name__}")
            raise
