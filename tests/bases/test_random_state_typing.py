import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from skfp.fingerprints import AtomPairFingerprint, MAPFingerprint
from skfp.model_selection import FingerprintEstimatorRandomizedSearch
from skfp.model_selection.splitters.randomized_scaffold_split import (
    randomized_scaffold_train_test_split,
)


@pytest.mark.parametrize(
    "random_state",
    [42, np.random.RandomState(42), None],
    ids=["int", "RandomState", "None"],
)
def test_map_fp_random_state_types(smallest_smiles_list, random_state):
    """MAPFingerprint should accept int, RandomState, or None."""
    fp = MAPFingerprint(random_state=random_state, n_jobs=-1)
    X = fp.transform(smallest_smiles_list)
    assert X.shape[0] == len(smallest_smiles_list)


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


@pytest.mark.parametrize(
    "random_state",
    [42, np.random.RandomState(42), None],
    ids=["int", "RandomState", "None"],
)
def test_randomized_scaffold_split_random_state_types(random_state):
    """randomized_scaffold_train_test_split should accept int, RandomState, or None."""
    smiles = [
        "C1CCCC(C2CC2)CC1",
        "c1n[nH]cc1C1CCCCCC1",
        "c1n[nH]cc1CC1CCCCCC1",
        "C1CCCC(CC2CCOCC2)CC1",
        "c1ccc2nc(OC3CCC3)ccc2c1",
        "O=C(CCc1cscn1)NC1CCNCC1",
        "c1ccc2nc(OC3CCOC3)ccc2c1",
        "c1ccc2nc(NC3CCOCC3)ccc2c1",
        "c1ccc2nc(N3CCCOCC3)ccc2c1",
        "c1ccc2nc(N3CCn4ccnc4C3)ccc2c1",
    ]
    train, test = randomized_scaffold_train_test_split(
        smiles, random_state=random_state
    )
    assert len(train) + len(test) == len(smiles)
