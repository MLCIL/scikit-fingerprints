import numpy as np
from numpy.testing import assert_allclose, assert_equal

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint


def test_new_mordred_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    mordred_fp = MordredFingerprint(n_jobs=-1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporarily prune X_old to test only against implemented descriptors in X_new
    effective_test_len = X_new.shape[1]
    X_old = X_old[:, :effective_test_len]

    assert_allclose(X_new, X_old, equal_nan=True)
    assert_equal(
        X_new.shape, (len(smallest_smiles_list), effective_test_len)
    )  # replace later to 1613
    assert X_new.dtype == np.float32


def test_new_mordred_sparse_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    mordred_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporarily prune X_old to test only against implemented descriptors in X_new
    effective_test_len = X_new.shape[1]
    X_old = X_old[:, :effective_test_len]

    assert_allclose(X_new.data, X_old.data, equal_nan=True)
    assert_equal(
        X_new.shape, (len(smallest_smiles_list), effective_test_len)
    )  # replace later to 1613
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporarily prune X_old to test only against implemented descriptors in X_new
    effective_test_len = X_new.shape[1]
    X_old = X_old[:, :effective_test_len]

    assert_allclose(X_new, X_old, equal_nan=True)
    assert_equal(
        X_new.shape, (len(smallest_smiles_list), effective_test_len)
    )  # replace later to 1826
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporarily prune X_old to test only against implemented descriptors in X_new
    effective_test_len = X_new.shape[1]
    X_old = X_old[:, :effective_test_len]

    assert_allclose(X_new.data, X_old.data, equal_nan=True)
    assert_equal(
        X_new.shape, (len(smallest_smiles_list), effective_test_len)
    )  # replace later to 1826
    assert X_new.dtype == np.float32


def test_new_mordred_feature_names():
    new_mordred_fp = NewMordredFingerprint()
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint()
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    assert_equal(feature_names_new, feature_names_old)


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    assert_equal(feature_names_new, feature_names_old)
