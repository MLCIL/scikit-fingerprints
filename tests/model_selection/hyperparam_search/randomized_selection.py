import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from skfp.fingerprints import AtomPairFingerprint
from skfp.model_selection import FingerprintEstimatorRandomizedSearch


class MockClassifier(DummyClassifier):
    def __init__(self, param1: int = 0, param2: int = 0):
        super().__init__(strategy="constant", constant=1)
        self.param1 = param1
        self.param2 = param2


def test_fp_estimator_randomized_search(smallest_mols_list):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))

    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": list(range(2, 31))}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorRandomizedSearch(
        fp, fp_param_grid, estimator_cv, n_iter=10
    )
    fp_cv.fit(smallest_mols_list, y)

    y_pred = fp_cv.predict(smallest_mols_list)
    y_pred_proba = fp_cv.predict_proba(smallest_mols_list)
    assert_equal(len(y_pred), len(smallest_mols_list))
    assert_equal(len(y_pred), len(y_pred_proba))

    X = fp_cv.transform(smallest_mols_list)
    assert_equal(X.shape, (len(smallest_mols_list), fp.n_features_out))

    assert_equal(len(fp_cv.cv_results_), 10)
    assert isinstance(fp_cv.best_fp_, AtomPairFingerprint)
    assert isinstance(fp_cv.best_fp_params_, dict)
    assert_allclose(fp_cv.best_score_, len(pos_mols) / num_mols)

    assert isinstance(fp_cv.best_estimator_cv_.best_estimator_, MockClassifier)
    assert_equal(fp_cv.best_estimator_cv_.best_params_, {"param1": 0, "param2": 0})


def test_fp_estimator_randomized_search_verbose(smallest_mols_list, capsys):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))

    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": list(range(2, 31))}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorRandomizedSearch(
        fp, fp_param_grid, estimator_cv, n_iter=10, verbose=3
    )
    fp_cv.fit(smallest_mols_list, y)

    output = capsys.readouterr().out
    assert output.startswith("Fitting 10 candidate hyperparameter sets")
    for i in range(1, 11):
        assert f"[{i}/10] START max_distance=" in output
        assert f"[{i}/10] END max_distance=" in output


def test_best_fp_caching(smallest_mols_list):
    num_mols = len(smallest_mols_list)
    pos_mols = np.ones(num_mols // 2)
    neg_mols = np.zeros(num_mols - len(pos_mols))
    y = np.concatenate((pos_mols, neg_mols))

    fp = AtomPairFingerprint()
    fp_param_grid = {"max_distance": [2, 3, 4]}
    estimator_cv = GridSearchCV(
        estimator=MockClassifier(),
        param_grid={"param1": [0, 1, 2], "param2": [0, 1, 2]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorRandomizedSearch(
        fp,
        fp_param_grid,
        estimator_cv,
        cache_best_fp_array=True,
        n_iter=10,
        verbose=3,
    )
    fp_cv.fit(smallest_mols_list, y)

    assert fp_cv.best_fp_array_ is not None
    assert isinstance(fp_cv.best_fp_array_, np.ndarray)
    assert_equal(fp_cv.best_fp_array_.shape, (num_mols, fp.n_features_out))


@pytest.mark.parametrize(
    "random_state",
    [42, np.random.RandomState(42), None],
    ids=["int", "RandomState", "None"],
)
def test_randomized_search_random_state_types(smallest_mols_list, random_state):
    """FingerprintEstimatorRandomizedSearch should accept int, RandomState, or None."""
    num_mols = len(smallest_mols_list)
    y = np.concatenate([np.ones(num_mols // 2), np.zeros(num_mols - num_mols // 2)])

    fp = AtomPairFingerprint()
    fp_params = {"max_distance": list(range(2, 6))}
    estimator_cv = GridSearchCV(
        estimator=DummyClassifier(strategy="constant", constant=1),
        param_grid={"constant": [0, 1]},
        scoring="accuracy",
    )
    fp_cv = FingerprintEstimatorRandomizedSearch(
        fp, fp_params, estimator_cv, n_iter=2, random_state=random_state
    )
    fp_cv.fit(smallest_mols_list, y)
    assert len(fp_cv.cv_results_) == 2
