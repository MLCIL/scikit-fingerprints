import numpy as np
from numpy.testing import assert_allclose, assert_equal

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint


def test_new_mordred_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    # temporary mask - will be eventually removed
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0)

    feature_names = new_mordred_fp.get_feature_names_out()
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_sparse_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    X_new = X_new.toarray()
    X_old = X_old.toarray()

    # temporary mask - will be eventually removed
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0)

    feature_names = new_mordred_fp.get_feature_names_out()
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    # temporary mask - will be eventually removed
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0)

    feature_names = new_mordred_fp.get_feature_names_out()
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    X_new = X_new.toarray()
    X_old = X_old.toarray()

    # temporary mask - will be eventually removed
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0)

    feature_names = new_mordred_fp.get_feature_names_out()
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_feature_names():
    new_mordred_fp = NewMordredFingerprint()
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint()
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # we exclude changed feature names
    changed_name = np.array(
        [
            "autocorr" in feature
            or "MoRSE" in feature
            or "BCUT" in feature
            or "Dz" in feature
            for feature in feature_names_new
        ]
    )
    feature_names_new = feature_names_new[~changed_name]
    feature_names_old = feature_names_old[~changed_name]

    assert_equal(feature_names_new, feature_names_old)


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # we exclude changed feature names
    changed_name = np.array(
        [
            "autocorr" in feature
            or "MoRSE" in feature
            or "BCUT" in feature
            or "Dz" in feature
            for feature in feature_names_new
        ]
    )
    feature_names_new = feature_names_new[~changed_name]
    feature_names_old = feature_names_old[~changed_name]

    assert_equal(feature_names_new, feature_names_old)


def _mismatched_features_msg(
    X_new: np.ndarray,
    X_old: np.ndarray,
    mask: np.ndarray,
    feature_names: np.ndarray,
    atol: float,
) -> str:
    # build an error message listing the feature names whose values
    # differ beyond the tolerance
    close = np.isclose(X_new[:, mask], X_old[:, mask], equal_nan=True, atol=atol)
    mismatched_cols = ~close.all(axis=0)
    mismatched_names = feature_names[mask][mismatched_cols]
    if len(mismatched_names) == 0:
        return ""
    return "Mismatched feature names:\n" + "\n".join(mismatched_names)
